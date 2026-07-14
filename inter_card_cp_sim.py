# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""
Inter-card CP 单进程模拟器（S0）—— 独立、自包含、不污染现有代码。

用途：在单进程里验证「两级 blocked scan == 单卡一趟平铺扫描」。
对应设计文档 docs/inter_card_cp.md §7/§8。

本文件当前实现 **L0（纯数学层）**：
  - 不依赖任何 tilelang / 已编译内核，纯 torch（默认 fp64），可在 CPU 运行。
  - 用抽象线性递推 h_{c+1} = M_c @ h_c + b_c（对应 GDN 的 inter-chunk 递推，
    见 tests/ref_gdr.py:torch_chunk_gdr_fwd）验证三级层次扫描的代数与边界记账：
        chunk --(aggregate, 模拟 prepare_h)--> sub-seq
        sub-seq --(reduce_local, 阶段 1a)-----> rank
        rank --(inter_scan, 阶段 1b)----------> global
        rank --(intra scan / correct, 阶段二)--> sub-seq h0
        sub-seq --(expand, 模拟 fused_fwd)-----> chunk h
  - L0 里所有 block 都是 fallback=True（精确），不含 warmup 近似——warmup/fallback
    的正确性属于 L2（真实内核）范畴。

后续（需 GPU + 编译内核）：
  - L1：torch reduce_local 与 tilelang correct_initial_states 的链式语义一致性。
  - L2：simulate_cp_forward 端到端 vs 单卡 chunk_gated_delta_rule 的 golden 对齐。

运行：  python inter_card_cp_sim.py
"""

from __future__ import annotations

import torch

# L0 用 fp64 做紧公差精确对拍
DTYPE = torch.float64
DEVICE = "cpu"


# ============================================================================
# S2 torch 原语（暂放本文件；S2/S3 真集成时抽进 cp/scan.py）
#
# 线性递推 block = (M, b)，语义 state_out = M @ state_in + b。
# 结合律（a 先、b 后）：
#     combine(a, b) = (M_b @ M_a,  M_b @ b_a + b_b)
# 单位元：(I, 0)。所有张量形状按 head 批处理：M[H,K,K], b[H,K,V]。
# ============================================================================


def combine(M_a, b_a, M_b, b_b):
    """把 block a（时间在前）与 block b（时间在后）合成一个等效 block。"""
    M = M_b @ M_a
    b = M_b @ b_a + b_b
    return M, b


def aggregate_blocks(Ms, bs, fallback=None):
    """把一串按时间顺序的 block 折叠成一个聚合 block (M_agg, b_agg)。

    模拟：chunk -> sub-seq（prepare_h 对 chunk 折叠）以及
          sub-seq -> rank（reduce_local 对 sub-seq 折叠）。

    Args:
        Ms: [P, H, K, K]  各 block 的转移矩阵
        bs: [P, H, K, V]  各 block 的偏置
        fallback: [P, H] bool 或 None。None 等价全 True（精确）。
                  False（强衰减）时该 block 视 incoming 为 0（M 项置零），
                  但自身 b 仍累加——对应 §7.3 的归零性质。
    Returns:
        (M_agg[H,K,K], b_agg[H,K,V])
    """
    P, H, K, _ = Ms.shape
    V = bs.shape[-1]
    M = torch.eye(K, dtype=Ms.dtype, device=Ms.device).expand(H, K, K).clone()
    b = torch.zeros(H, K, V, dtype=bs.dtype, device=bs.device)
    for i in range(P):
        Mi = Ms[i]
        if fallback is not None:
            # 非 fallback：incoming 被本 block 杀掉 -> Mi 在 M 链与 b 传播中置零
            mask = fallback[i].to(Ms.dtype).view(H, 1, 1)  # [H,1,1]
            Mi = Mi * mask
        b = bs[i] + Mi @ b
        M = Mi @ M
    return M, b


def reduce_local(mt, ht, seq_reset, fallback=None):  # noqa: F811 (thin re-export)
    """从生产模块导入的 S2 原语（见 flash_qla/.../chunk/cp/scan.py）。"""
    from flash_qla.ops.gated_delta_rule.chunk.cp import reduce_local as _reduce_local
    return _reduce_local(mt, ht, seq_reset, fallback)


def inter_scan(neigh_M, neigh_S):  # noqa: F811 (thin re-export)
    """从生产模块导入的 S2 原语（见 flash_qla/.../chunk/cp/scan.py）。"""
    if neigh_M.shape[0] == 0:
        H, K, _ = neigh_M.shape[1:]
        V = neigh_S.shape[-1]
        return torch.zeros(H, K, V, dtype=neigh_M.dtype, device=neigh_M.device)
    from flash_qla.ops.gated_delta_rule.chunk.cp import inter_scan as _inter_scan
    return _inter_scan(neigh_M, neigh_S)


def intra_scan(mt, ht, seq_reset, seed, fallback=None):
    """阶段二（correct 的抽象版）：以 seed 为首条序列种子，沿本卡 sub-seq 做 prefix scan。

    返回每个 sub-seq 的 h0（进入该 sub-seq 前的状态）。首条序列（第一个 reset 段）
    用 seed 起算，其余 raw 序列段从 0 起算。
    """
    P, H, K, _ = mt.shape
    V = ht.shape[-1]
    h0 = torch.zeros(P, H, K, V, dtype=ht.dtype, device=ht.device)
    S = None
    seen_first = False
    for i in range(P):
        if seq_reset[i]:
            if not seen_first:
                S = seed.clone()
                seen_first = True
            else:
                S = torch.zeros(H, K, V, dtype=ht.dtype, device=ht.device)
        h0[i] = S
        Mi = mt[i]
        if fallback is not None:
            Mi = Mi * fallback[i].to(mt.dtype).view(H, 1, 1)
        S = ht[i] + Mi @ S
    return h0


# ============================================================================
# 分区元数据（模拟 build_cp_context + _calc_cp_seqs），chunk 粒度
# ============================================================================


def rank_of_chunk(c, part, W):
    """chunk c 属于哪张卡（余数并入最后一张卡）。"""
    return min(c // part, W - 1)


def build_partition(raw_seqlens, W, max_local_chunks):
    """按 chunk 粒度构造两级分区元数据。

    Args:
        raw_seqlens: list[int]  各 raw 序列的 chunk 数（varlen）
        W: int  卡数
        max_local_chunks: int  卡内 sub-seq 的最大 chunk 数

    Returns:
        dict，包含：
          total_chunks, part, chunk_seq_id[total], raw_seq_start[num_seq]
          ranks: list（每张卡一项）：
            {lo, hi, subseqs:[(c_lo,c_hi,seq_id)], seq_reset:[bool],
             first_seq_id, pre_num_ranks, post_num_ranks, is_first_rank, is_last_rank}
    """
    total = sum(raw_seqlens)
    assert total >= W, "total chunks 必须 >= W"
    part = total // W
    assert part >= 1

    # 每个 chunk 的 raw 序列 id，以及各 raw 序列的全局起始 chunk
    chunk_seq_id = []
    raw_seq_start = []
    acc = 0
    for sid, L in enumerate(raw_seqlens):
        raw_seq_start.append(acc)
        chunk_seq_id.extend([sid] * L)
        acc += L

    def rk(c):
        return rank_of_chunk(c, part, W)

    ranks = []
    for r in range(W):
        lo = r * part
        hi = total if r == W - 1 else (r + 1) * part
        # 卡内按 raw 序列切段，再按 max_local_chunks 切 sub-seq
        subseqs = []          # (c_lo, c_hi, seq_id)
        seq_reset = []        # 该 sub-seq 是否起一条新 raw 序列（卡内视角）
        c = lo
        while c < hi:
            sid = chunk_seq_id[c]
            # 本段：同一 raw 序列且在 [lo,hi) 内的最大连续跨度
            seg_lo = c
            while c < hi and chunk_seq_id[c] == sid:
                c += 1
            seg_hi = c
            first_sub_in_seg = True
            s = seg_lo
            while s < seg_hi:
                e = min(s + max_local_chunks, seg_hi)
                subseqs.append((s, e, sid))
                seq_reset.append(first_sub_in_seg)
                first_sub_in_seg = False
                s = e

        first_seq_id = chunk_seq_id[lo]
        first_rank_of_first_seq = rk(raw_seq_start[first_seq_id])
        pre_num_ranks = r - first_rank_of_first_seq
        is_first_rank = pre_num_ranks == 0

        last_seq_id = chunk_seq_id[hi - 1]
        last_seq_end_chunk = raw_seq_start[last_seq_id] + raw_seqlens[last_seq_id] - 1
        last_rank_of_last_seq = rk(last_seq_end_chunk)
        post_num_ranks = last_rank_of_last_seq - r
        is_last_rank = post_num_ranks == 0

        ranks.append(dict(
            lo=lo, hi=hi, subseqs=subseqs, seq_reset=seq_reset,
            first_seq_id=first_seq_id, pre_num_ranks=pre_num_ranks,
            post_num_ranks=post_num_ranks,
            is_first_rank=is_first_rank, is_last_rank=is_last_rank,
        ))

    return dict(total_chunks=total, part=part, chunk_seq_id=chunk_seq_id,
                raw_seq_start=raw_seq_start, raw_seqlens=list(raw_seqlens),
                ranks=ranks)


# ============================================================================
# 参考（单卡一趟平铺扫描）
# ============================================================================


def flat_scan(chunk_M, chunk_b, chunk_seq_id):
    """平铺参考：逐 chunk 递推，raw 序列边界处重置。返回每个 chunk 的进入状态 h。"""
    N, H, K, _ = chunk_M.shape
    V = chunk_b.shape[-1]
    S = torch.zeros(H, K, V, dtype=chunk_M.dtype, device=chunk_M.device)
    h = torch.zeros(N, H, K, V, dtype=chunk_M.dtype, device=chunk_M.device)
    for c in range(N):
        if c == 0 or chunk_seq_id[c] != chunk_seq_id[c - 1]:
            S = torch.zeros(H, K, V, dtype=chunk_M.dtype, device=chunk_M.device)
        h[c] = S
        S = chunk_M[c] @ S + chunk_b[c]
    return h


# ============================================================================
# 层次扫描（用上面的 S2 原语），模拟 W 卡
# ============================================================================


def hierarchical_scan(chunk_M, chunk_b, meta):
    """三级层次扫描：chunk->subseq(aggregate) -> rank(reduce) -> global(inter_scan)
       -> subseq h0(intra_scan) -> chunk h(expand)。返回每个 chunk 的进入状态 h。"""
    N, H, K, _ = chunk_M.shape
    V = chunk_b.shape[-1]
    ranks = meta["ranks"]
    W = len(ranks)

    # --- 步骤1 + 阶段1a：每卡对 chunk 折叠成 sub-seq，再 reduce 成卡聚合 ---
    per_rank_sub = []   # 每卡：(mt[P,H,K,K], ht[P,H,K,V], seq_reset[P])
    rank_M = torch.zeros(W, H, K, K, dtype=chunk_M.dtype, device=chunk_M.device)
    rank_S = torch.zeros(W, H, K, V, dtype=chunk_b.dtype, device=chunk_b.device)
    for r, rk in enumerate(ranks):
        subseqs = rk["subseqs"]
        P = len(subseqs)
        mt = torch.zeros(P, H, K, K, dtype=chunk_M.dtype, device=chunk_M.device)
        ht = torch.zeros(P, H, K, V, dtype=chunk_b.dtype, device=chunk_b.device)
        for i, (clo, che, _sid) in enumerate(subseqs):
            Magg, bagg = aggregate_blocks(chunk_M[clo:che], chunk_b[clo:che])
            mt[i], ht[i] = Magg, bagg
        seq_reset = rk["seq_reset"]
        per_rank_sub.append((mt, ht, seq_reset))
        M_r, S_ext_r = reduce_local(mt, ht, seq_reset)
        rank_M[r], rank_S[r] = M_r, S_ext_r

    # --- 阶段1b：跨卡 inter_scan 得每卡 incoming initial_state_r ---
    # （L0 用完整 rank_M/rank_S 数组模拟 all_gather）
    h = torch.zeros(N, H, K, V, dtype=chunk_b.dtype, device=chunk_b.device)
    for r, rk in enumerate(ranks):
        pre = rk["pre_num_ranks"]
        if pre == 0:
            seed = torch.zeros(H, K, V, dtype=chunk_b.dtype, device=chunk_b.device)
        else:
            neigh_M = rank_M[r - pre:r]     # [pre,H,K,K] 从前往后
            neigh_S = rank_S[r - pre:r]
            seed = inter_scan(neigh_M, neigh_S)

        # --- 阶段二：卡内 intra_scan 得每 sub-seq h0 ---
        mt, ht, seq_reset = per_rank_sub[r]
        sub_h0 = intra_scan(mt, ht, seq_reset, seed)

        # --- expand：sub-seq h0 -> 每 chunk 进入状态 ---
        for i, (clo, che, _sid) in enumerate(rk["subseqs"]):
            S = sub_h0[i].clone()
            for c in range(clo, che):
                h[c] = S
                S = chunk_M[c] @ S + chunk_b[c]

    return h


# ============================================================================
# 随机数据生成
# ============================================================================


def make_data(raw_seqlens, H, K, V, decay, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    N = sum(raw_seqlens)
    R = torch.randn(N, H, K, K, generator=g, dtype=DTYPE, device=DEVICE) / (K ** 0.5)
    chunk_M = decay * R                      # 控制谱半径 ~ decay
    chunk_b = torch.randn(N, H, K, V, generator=g, dtype=DTYPE, device=DEVICE)
    chunk_seq_id = []
    for sid, L in enumerate(raw_seqlens):
        chunk_seq_id.extend([sid] * L)
    return chunk_M, chunk_b, chunk_seq_id


# ============================================================================
# L0 测试矩阵
# ============================================================================


def run_case(name, raw_seqlens, W, max_local_chunks, decay, H=3, K=8, V=8, seed=0):
    chunk_M, chunk_b, chunk_seq_id = make_data(raw_seqlens, H, K, V, decay, seed)
    meta = build_partition(raw_seqlens, W, max_local_chunks)

    # 内部断言：分区元数据自洽
    assert meta["chunk_seq_id"] == chunk_seq_id
    assert sum(rk["hi"] - rk["lo"] for rk in meta["ranks"]) == meta["total_chunks"]
    for r in range(W - 1):
        # rank r 的 post 与 rank r+1 的 pre 应对应同一条边界序列
        rk, nxt = meta["ranks"][r], meta["ranks"][r + 1]
        shares = (chunk_seq_id[rk["hi"] - 1] == chunk_seq_id[nxt["lo"]])
        assert (rk["post_num_ranks"] > 0) == shares == (nxt["pre_num_ranks"] > 0), name

    h_ref = flat_scan(chunk_M, chunk_b, chunk_seq_id)
    h_hier = hierarchical_scan(chunk_M, chunk_b, meta)

    rel = (h_hier - h_ref).norm() / (h_ref.norm() + 1e-30)
    ok = torch.allclose(h_hier, h_ref, rtol=1e-9, atol=1e-9)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:52s} W={W} mlc={max_local_chunks} decay={decay:<4} "
          f"seqs={raw_seqlens} rel={rel.item():.2e}")
    return ok


def main():
    print("=" * 100)
    print("L0 纯数学层次扫描验证（fp64，精确对拍；两级 blocked scan == 单卡一趟）")
    print("=" * 100)
    results = []

    for decay in (0.5, 0.95, 1.0):  # 强衰减 / 弱衰减 / 无衰减
        # (a) 一条长序列跨全部 W 卡
        results.append(run_case("(a) 单长序列跨全卡", [64], 4, 3, decay))
        # (b) 多序列、边界对齐卡界（每卡整条序列，无跨卡）
        results.append(run_case("(b) 多序列边界对齐", [16, 16, 16, 16], 4, 3, decay))
        # (c) 多序列、边界不对齐（跨卡边界落在序列内部）
        results.append(run_case("(c) 多序列边界不对齐", [20, 13, 27], 4, 4, decay))
        # (d) 一条序列跨 >2 卡（pre/post_num_ranks>1）
        results.append(run_case("(d) 单序列跨>2卡", [96], 6, 5, decay))
        # (e) part 不整除（余数并入末卡）
        results.append(run_case("(e) part不整除", [50, 17], 4, 3, decay))
        # W=1 退化
        results.append(run_case("(f) W=1 退化", [40, 24], 1, 3, decay))
        # sub-seq 只有 1 chunk（mlc=1）
        results.append(run_case("(g) mlc=1", [30, 34], 4, 1, decay))

    print("-" * 100)
    n_pass = sum(results)
    print(f"总计 {n_pass}/{len(results)} 通过")
    if n_pass != len(results):
        raise SystemExit(1)


# ============================================================================
# L2：真实内核端到端模拟器（需 GPU + 已编译 FlashQLA 内核）
#
# 每卡用精确聚合路线：fused_gdr_h(全 warmup) 得本卡各 raw 序列的精确 (ht, mt)，
# 取末条序列作为跨卡聚合 (M_r, S_ext_r)；inter_scan 还原本卡首序列 incoming；
# 以其为 initial_state 跑真实 chunk_gated_delta_rule（intra auto_cp 可开可关）。
# 拼接输出 vs 单卡 golden。
# ============================================================================

import bisect

L2_DTYPE = torch.bfloat16
CHUNK = 64


def _seq_of_token(t, cu_global):
    # 返回 token t 所属 raw 序列 index：最大 i 使 cu_global[i] <= t
    return bisect.bisect_right(cu_global, t) - 1


def _rank_of_token(t, part, W):
    return min(t // part, W - 1)


def build_token_partition(cu_global, W):
    """token 粒度的两级 inter 分区（模拟 build_cp_context）。要求 T%W==0 且 part%CHUNK==0。"""
    T = cu_global[-1]
    assert T % W == 0, f"T={T} 必须能被 W={W} 整除"
    part = T // W
    assert part % CHUNK == 0, f"part={part} 需为 chunk({CHUNK}) 对齐"

    ranks = []
    for r in range(W):
        lo = r * part
        hi = (r + 1) * part
        pts = [b for b in cu_global if lo < b < hi]
        local_ends = [lo] + pts + [hi]
        local_cu = [x - lo for x in local_ends]  # 本地坐标，起于 0

        first_seq_id = _seq_of_token(lo, cu_global)
        first_seq_start = cu_global[first_seq_id]
        first_rank = _rank_of_token(first_seq_start, part, W)
        pre = r - first_rank

        last_seq_id = _seq_of_token(hi - 1, cu_global)
        last_seq_end = cu_global[last_seq_id + 1] - 1
        last_rank = _rank_of_token(last_seq_end, part, W)
        post = last_rank - r

        ranks.append(dict(lo=lo, hi=hi, local_cu=local_cu,
                          pre_num_ranks=pre, post_num_ranks=post,
                          is_first_rank=(pre == 0), is_last_rank=(post == 0)))
    return dict(T=T, part=part, ranks=ranks)


def _rank_exact_aggregate(k_r, v_r, g_r, beta_r, local_cu, scale):
    """本卡各 raw 序列的精确 (ht, mt)：fused_gdr_h 全 warmup。取末条序列做跨卡聚合。"""
    from flash_qla.ops.utils import chunk_local_cumsum
    from flash_qla.ops.gated_delta_rule.chunk import kkt_solve, fused_gdr_h

    N = len(local_cu) - 1
    Hv = v_r.shape[2]
    cul = torch.tensor(local_cu, device=k_r.device, dtype=torch.int32)

    g_c = chunk_local_cumsum(g_r, cu_seqlens=cul, chunk_size=CHUNK)
    A = kkt_solve(k_r, beta_r, cu_seqlens=cul, chunk_size=CHUNK)

    seqlens = [local_cu[i + 1] - local_cu[i] for i in range(N)]
    nchunks = [(s + CHUNK - 1) // CHUNK for s in seqlens]
    num_warmup = torch.zeros(N, Hv, device=k_r.device, dtype=torch.int32)
    for i in range(N):
        num_warmup[i, :] = nchunks[i]  # 全 warmup -> 精确

    _, ht, mt = fused_gdr_h(
        k=k_r, v=v_r, a=A, g=g_c, b=beta_r,
        initial_state=None, output_final_state=True, output_h=False,
        cu_seqlens=cul, num_warmup_chunks=num_warmup,
    )
    # 末条序列聚合
    return mt[-1].float(), ht[-1].float()  # M_r[Hv,K,K], S_ext_r[Hv,K,V]


def _rank_subseq_buffers(k_r, v_r, g_r, beta_r, local_cu, force_full=False):
    """本卡 warmup 路径的 sub-seq 缓冲（复刻 intra_card_cp_preprocess 前半段）。

    force_full=False：用 get_warmup_chunks 的自动截断 warmup（intra 生产设置；**截断缓冲
        不能用于跨卡聚合**）。
    force_full=True ：强制每个 sub-seq 全量 warmup（全 fallback，精确 per-subseq (M,f)），
        此时 reduce_local 链乘 == 整段精确（诊断已证）。

    返回 dict(mt, ht[原始 bf16], fallback[P,Hv] bool, seq_reset[list], seq_map_r2c, cu)。
    """
    from flash_qla.ops.utils import chunk_local_cumsum
    from flash_qla.ops.gated_delta_rule.chunk import (
        kkt_solve, fused_gdr_h, _calc_cp_seqs, get_warmup_chunks)
    from flash_qla.ops.gated_delta_rule.chunk.cp import seq_reset_from_seq_map

    Hv = v_r.shape[2]
    cul = torch.tensor(local_cu, device=k_r.device, dtype=torch.int32)
    g_c = chunk_local_cumsum(g_r, cu_seqlens=cul, chunk_size=CHUNK)
    A = kkt_solve(k_r, beta_r, cu_seqlens=cul, chunk_size=CHUNK)

    use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask, ht_mask_bwd = _calc_cp_seqs(
        raw_cu_seqlens=cul, chunk_size=CHUNK, num_v_heads=Hv, is_bwd=False)

    if not use_cp:
        N = len(local_cu) - 1
        seqlens = [local_cu[i + 1] - local_cu[i] for i in range(N)]
        nchunks = [(s + CHUNK - 1) // CHUNK for s in seqlens]
        num_warmup = torch.zeros(N, Hv, device=k_r.device, dtype=torch.int32)
        for i in range(N):
            num_warmup[i, :] = nchunks[i]
        _, ht, mt = fused_gdr_h(
            k=k_r, v=v_r, a=A, g=g_c, b=beta_r, initial_state=None,
            output_final_state=True, output_h=False,
            cu_seqlens=cul, num_warmup_chunks=num_warmup)
        return dict(mt=mt, ht=ht,
                    fallback=torch.ones(N, Hv, dtype=torch.bool, device=k_r.device),
                    seq_reset=[True] * N,
                    seq_map_r2c=torch.arange(N + 1, device=k_r.device), cu=cul)

    P = cp_cu_seqlens.shape[0] - 1
    if force_full:
        # 每个 sub-seq 强制全量 warmup -> 精确 per-subseq (M,f)，全 fallback
        cp_list = cp_cu_seqlens.tolist()
        nchunks = [(cp_list[i + 1] - cp_list[i] + CHUNK - 1) // CHUNK for i in range(P)]
        num_warmup = torch.zeros(P, Hv, device=k_r.device, dtype=torch.int32)
        for i in range(P):
            num_warmup[i, :] = nchunks[i]
        fallback = torch.ones(P, Hv, dtype=torch.bool, device=k_r.device)
    else:
        num_warmup, fb = get_warmup_chunks(
            g=g_c, cu_seqlens=cp_cu_seqlens, ht_mask=ht_mask, chunk_size=CHUNK, threshold=-10.0)
        fallback = fb.bool()
    _, ht, mt = fused_gdr_h(
        k=k_r, v=v_r, a=A, g=g_c, b=beta_r, initial_state=None,
        output_final_state=True, output_h=False,
        cu_seqlens=cp_cu_seqlens, num_warmup_chunks=num_warmup)
    return dict(mt=mt, ht=ht, fallback=fallback,
                seq_reset=seq_reset_from_seq_map(seq_map_r2c, P),
                seq_map_r2c=seq_map_r2c, cu=cp_cu_seqlens)


def _build_rank_contexts(cu_global, W, device):
    """用生产 S1 `get_cp_cu_seqlens` 逐 rank 构造上下文（单进程，group=None）。

    返回 list[dict]，键与 build_token_partition 对齐：lo/hi/local_cu/pre_num_ranks/
    post_num_ranks/is_first_rank/is_last_rank。
    """
    from flash_qla.ops.gated_delta_rule.chunk.cp import get_cp_cu_seqlens
    T = cu_global[-1]
    assert T % W == 0
    part = T // W
    cul = torch.tensor(cu_global, device=device, dtype=torch.int32)
    ranks = []
    for r in range(W):
        ctx = get_cp_cu_seqlens(cul, world_size=W, rank=r, group=None)
        ranks.append(dict(
            lo=r * part, hi=(r + 1) * part,
            local_cu=ctx.cu_seqlens_cpu.tolist(),
            pre_num_ranks=ctx.pre_num_ranks, post_num_ranks=ctx.post_num_ranks,
            is_first_rank=ctx.is_first_rank, is_last_rank=ctx.is_last_rank))
    return ranks


def simulate_cp_forward(q, k, v, g, beta, cu_global, W, scale, intra_auto_cp,
                        agg_mode="subseq_full"):
    """单进程模拟 W 卡 inter-card CP 前向，返回拼接后的全局输出 [1, T, Hv, V]。

    分区用生产 S1 `get_cp_cu_seqlens`；跨卡聚合用 S2 `reduce_local`/`inter_scan`；主前向用真实内核。
    这是**模拟下的完整 inter-card CP**（all_gather 用 list 堆叠模拟，S3 换成真 dist 后应逐位一致）。

    agg_mode（阶段 1a 如何得跨卡聚合 (M_r, S_ext_r)）见 _rank_subseq_buffers/exact 说明。
    """
    from flash_qla.ops.gated_delta_rule import chunk_gated_delta_rule

    ranks = _build_rank_contexts(cu_global, W, k.device)   # S1
    Hv, K, V = v.shape[2], k.shape[3], v.shape[3]

    # --- 阶段一 1a：各卡聚合 (M_r, S_ext_r) ---
    rank_M = torch.zeros(W, Hv, K, K, device=k.device, dtype=torch.float32)
    rank_S = torch.zeros(W, Hv, K, V, device=k.device, dtype=torch.float32)
    for r, rk in enumerate(ranks):
        lo, hi = rk["lo"], rk["hi"]
        if agg_mode == "exact":
            M_r, S_ext_r = _rank_exact_aggregate(
                k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi], rk["local_cu"], scale)
        else:
            buf = _rank_subseq_buffers(
                k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi], rk["local_cu"],
                force_full=(agg_mode == "subseq_full"))
            M_r, S_ext_r = reduce_local(
                buf["mt"].float(), buf["ht"].float(), buf["seq_reset"], buf["fallback"])
        rank_M[r], rank_S[r] = M_r, S_ext_r

    # --- 阶段一 1b + 真实前向（阶段二在 chunk_gated_delta_rule 内部完成）---
    outs = []
    for r, rk in enumerate(ranks):
        lo, hi = rk["lo"], rk["hi"]
        N_r = len(rk["local_cu"]) - 1
        pre = rk["pre_num_ranks"]
        raw_h0 = torch.zeros(N_r, Hv, K, V, device=k.device, dtype=torch.float32)
        if pre > 0:
            raw_h0[0] = inter_scan(rank_M[r - pre:r], rank_S[r - pre:r])
        cul = torch.tensor(rk["local_cu"], device=k.device, dtype=torch.int32)
        o_r, _ = chunk_gated_delta_rule(
            q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi],
            scale=scale, initial_state=raw_h0, cu_seqlens=cul,
            output_final_state=True, auto_cp=intra_auto_cp,
        )
        outs.append(o_r)
    return torch.cat(outs, dim=1)


def _make_l2_data(cu_global, Hk, Hv, seed=42):
    import torch.nn.functional as F
    torch.manual_seed(seed)
    T = cu_global[-1]
    K = V = 128
    dev = "cuda"
    q = F.normalize(torch.randn(1, T, Hk, K, device=dev, dtype=L2_DTYPE), p=2, dim=-1)
    k = F.normalize(torch.randn(1, T, Hk, K, device=dev, dtype=L2_DTYPE), p=2, dim=-1)
    v = torch.randn(1, T, Hv, V, device=dev, dtype=L2_DTYPE)
    beta = torch.randn(1, T, Hv, device=dev, dtype=torch.float32).sigmoid()
    return q, k, v, beta


def _ratio(a, b):
    max_err = (a.float() - b.float()).abs().max().item()
    max_ref = b.float().abs().max().item()
    return max_err / (max_ref + 1e-30)


def run_l2_case(name, cu_global, W, Hk, Hv, g_scale, intra_auto_cp,
                agg_mode="warmup", rtol=2e-2, seed=42):
    import torch.nn.functional as F
    from flash_qla.ops.gated_delta_rule import chunk_gated_delta_rule

    q, k, v, beta = _make_l2_data(cu_global, Hk, Hv, seed)
    T = cu_global[-1]
    # g: log 空间衰减，g_scale 控制强弱（越小衰减越弱=跨卡越吃劲）
    g = F.logsigmoid(torch.randn(1, T, Hv, device="cuda", dtype=torch.float32)) * g_scale
    scale = k.shape[-1] ** -0.5
    cul_global = torch.tensor(cu_global, device="cuda", dtype=torch.int32)

    o_ref, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale, cu_seqlens=cul_global,
        output_final_state=True, auto_cp=False)

    o_cp = simulate_cp_forward(q, k, v, g, beta, cu_global, W, scale,
                               intra_auto_cp, agg_mode=agg_mode)

    r = _ratio(o_cp, o_ref)
    ok = r <= rtol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:36s} agg={agg_mode:6s} W={W} intra_cp={int(intra_auto_cp)} "
          f"g×{g_scale:<6} Hk/Hv={Hk}/{Hv} cu={cu_global} ratio={r:.2e}")
    return ok


def main_l2():
    if not torch.cuda.is_available():
        print("[L2] 跳过：无 CUDA")
        return True
    print("=" * 108)
    print("L2 真实内核端到端（inter-card CP 模拟 vs 单卡 golden；bf16 ratio<2e-2）")
    print("  agg=exact       : fla 风格独立精确一趟（fused_gdr_h over rank raw cu_seqlens）")
    print("  agg=subseq_full : 单趟共用 prepare_h + 全量 warmup + reduce_local（推荐设计）")
    print("=" * 108)
    results = []
    for agg in ("exact", "subseq_full"):
        for gs in (1.0 / 16, 1.0 / 4, 1.0):
            for cp in (False, True):
                results.append(run_l2_case("(a) 单长序列跨全卡", [0, 2048], 4, 2, 2, gs, cp, agg))
                results.append(run_l2_case("(c) 多序列含跨卡", [0, 512, 1536, 2048], 4, 2, 2, gs, cp, agg))
                results.append(run_l2_case("(d) 序列跨3卡", [0, 1536, 2048], 4, 2, 2, gs, cp, agg))
                results.append(run_l2_case("(e) GQA 跨全卡", [0, 2048], 4, 2, 4, gs, cp, agg))
    print("-" * 108)
    n = sum(results)
    print(f"L2 总计 {n}/{len(results)} 通过")
    return n == len(results)


def main_l2b_probe():
    """负面结果探针：用 intra 的**截断** warmup 缓冲（get_warmup_chunks）做跨卡聚合。

    结论（2026-07-12）：**不成立**。截断 warmup 只跑子序列尾部、缺前段贡献，链乘卡聚合会崩
    （即使弱衰减档，ratio~0.6）。对比 main_l2 的 agg=subseq_full（全量 warmup）则通过——
    说明问题在 warmup **长度**（截断），不在 reduce_local 本身。共用 prepare_h 时 inter-card
    需按自身尺度配足 warmup。见 docs/inter_card_cp.md §3。
    """
    if not torch.cuda.is_available():
        print("[L2b] 跳过：无 CUDA")
        return
    print("=" * 108)
    print("L2b 探针（截断 warmup + reduce_local，已知不成立；对照组，记录负面结果）")
    print("=" * 108)
    for gs in (1.0 / 16, 1.0):
        run_l2_case("(a) 单长序列跨全卡", [0, 2048], 4, 2, 2, gs, False, "warmup")


def main_l1():
    """L1：torch intra_scan（reduce_local 同款链式语义）逐位复刻 tilelang correct_initial_states。"""
    if not torch.cuda.is_available():
        print("[L1] 跳过：无 CUDA")
        return True
    from flash_qla.ops.gated_delta_rule.chunk import correct_initial_states
    print("=" * 108)
    print("L1 一致性（torch 链式 vs tilelang correct_initial_states；raw_h0=0）")
    print("=" * 108)
    results = []
    for gs in (1.0 / 16, 1.0 / 4, 1.0):
        # 用一条足够长的单序列切片，强制 use_cp=True 且多 sub-seq
        q, k, v, beta = _make_l2_data([0, 1024], 2, 2, seed=7)
        import torch.nn.functional as F
        g = F.logsigmoid(torch.randn(1, 1024, 2, device="cuda", dtype=torch.float32)) * gs
        buf = _rank_subseq_buffers(k, v, g, beta, [0, 1024])
        P = buf["mt"].shape[0]

        # tilelang correct（raw_h0=None -> 首 sub-seq h0=0）
        cp_h0_tl = correct_initial_states(
            raw_h0=None, ht_buffer=buf["ht"], mt_buffer=buf["mt"],
            fallback_mask=buf["fallback"], seq_map_r2c=buf["seq_map_r2c"])

        # torch 链式（seed=0），复用 reduce_local/correct 同款公式
        Hv, K, V = v.shape[2], k.shape[3], v.shape[3]
        seed0 = torch.zeros(Hv, K, V, device="cuda", dtype=torch.float32)
        h0_torch = intra_scan(buf["mt"].float(), buf["ht"].float(),
                              buf["seq_reset"], seed0, buf["fallback"])

        r = _ratio(h0_torch, cp_h0_tl.float())
        ok = r <= 2e-2
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] g×{gs:<6} P(sub-seq)={P} ratio={r:.2e}")
    print("-" * 108)
    n = sum(results)
    print(f"L1 总计 {n}/{len(results)} 通过")
    return n == len(results)


def main_s1():
    """S1 交叉校验：生产 get_cp_cu_seqlens 的 rank-local 元数据 == 模拟器 inline 分区。"""
    print("=" * 108)
    print("S1 交叉校验（build_cp_context / get_cp_cu_seqlens vs inline build_token_partition）")
    print("=" * 108)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cases = [
        ([0, 2048], 4), ([0, 512, 1536, 2048], 4), ([0, 1536, 2048], 4),
        ([0, 4096], 8), ([0, 1024, 2048, 3072, 4096], 4), ([0, 2048], 1),
    ]
    results = []
    for cu, W in cases:
        inline = build_token_partition(cu, W)["ranks"]
        ctxs = _build_rank_contexts(cu, W, torch.device(dev))
        ok = True
        for r in range(W):
            a, b = inline[r], ctxs[r]
            ok &= (a["local_cu"] == b["local_cu"]
                   and a["pre_num_ranks"] == b["pre_num_ranks"]
                   and a["post_num_ranks"] == b["post_num_ranks"]
                   and a["is_first_rank"] == b["is_first_rank"]
                   and a["is_last_rank"] == b["is_last_rank"])
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] cu={cu} W={W}")
    print("-" * 108)
    n = sum(results)
    print(f"S1 总计 {n}/{len(results)} 通过")
    return n == len(results)


if __name__ == "__main__":
    import sys
    main()
    ok = True
    if "--s1" in sys.argv or "--all" in sys.argv:
        ok = main_s1() and ok
    if "--l1" in sys.argv or "--all" in sys.argv:
        ok = main_l1() and ok
    if "--l2" in sys.argv or "--all" in sys.argv:
        ok = main_l2() and ok
    if "--l2b" in sys.argv:
        main_l2b_probe()   # 负面结果探针，不参与 gating
    if not ok:
        raise SystemExit(1)
