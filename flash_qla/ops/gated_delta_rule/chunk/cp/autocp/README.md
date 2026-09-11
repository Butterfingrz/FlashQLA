# FlashQLA Auto CP

[English](#english) · [中文](#中文)

---

## English

**Auto CP** decides *whether* to enable intra-card CP and, if so, *what
subsequence length* to use. It predicts the latency of a few candidate configs
with a lightweight kernel model and compares the best CP candidate against no-CP.

Auto CP has three steps:

1. **Kernel latency model** — predict each kernel's time from its launch shape.
2. **CP partition search** — evaluate only a handful of subsequence lengths near
   execution-wave boundaries.
3. **CP enablement** — pick the best CP candidate only if it beats no-CP by a margin.

### 1. Kernel latency model

Every GDN kernel repeats a fixed-granularity unit of work (a chunk-level
recurrence step, or a CP-partition op). We abstract a CTA's workload by its
work-unit count and fit one linear model per kernel row:
$t_\kappa = a \cdot d_\kappa + b \cdot u + c$, where:

- $d_\kappa$ — **critical schedule depth**: work units on the busiest
  SM under a greedy CTA-to-SM assignment. Computed by `schedule_depth()`
  (`utils.py`), which simulates greedy packing of CTA runs onto $P$ SMs and
  returns the makespan.
- $u = H \cdot n_{\mathrm{part}}$ — **CTA population** (total CTAs), the
  launch/scheduling overhead of more CTAs. The SM count $P$ is not divided out
  here; it is absorbed into the fitted $b$.
- $a, b, c$ — fitted per kernel family and per hardware arch.

With every kernel modeled, FlashQLA Auto CP supports two modes selected by
`is_train`. `infer` mode targets the summed time of all CP-related kernels in the
forward pass; `train` mode targets the summed time of all CP-related kernels
across both the forward and backward passes — for inference and training
respectively.

Fitted coefficients are stored in `coefs/<arch>.csv`; the mapping from each
kernel's `block_DV` to its coefficient row is resolved in `launch.py`.

### 2. CP partition search

Candidate subsequence lengths come from `wave_candidates()`. For a batch of $B$
sequences with lengths $S_b$, $H$ heads, and $P$ SMs, a subsequence length $L$
induces $N_{\mathrm{cp}}(L) = \sum_b \lceil S_b / L \rceil$ CP partitions and
approximately $w(L) = \lceil H \cdot N_{\mathrm{cp}} / P \rceil$ execution waves.
Latency changes most sharply where $w(L)$ crosses an integer, so Auto CP solves
for the $L$ at each wave boundary (for equal-length sequences $S$, the $L$ giving
$q$ waves is $L \approx H B S / (q P)$) and searches `MAX_EVALS` candidates in
descending order of $L$ to obtain the best CP partition.

### 3. CP enablement

Enabling CP introduces overhead, so CP is unnecessary when the workload already
exposes enough parallelism. Auto CP decides by comparing the best CP candidate's
time against the no-CP time: `use_cp` iff
$T_{\mathrm{cp}} < T_{\mathrm{nocp}} \cdot (1 - \mathrm{margin})$ (margin default
0.05).

### Usage

```python
from flash_qla.ops.gated_delta_rule.chunk.cp.autocp import decide

use_cp, lcp = decide(
    num_chunks=[...],      # or cu_seqlens=... / seq_lens=...
    num_v_heads=H,
    P=sm_count,            # defaults to the CSV's P
    chunk=chunk_size,      # defaults to the CSV's chunk
    is_train=False,
    g=gate_tensor,         # optional; enables warmup-aware prepare_h depth
)
# use_cp: bool, lcp: max local chunks (None when CP is off)
```

Wired into FlashQLA as the default intra-card CP partitioning strategy.

### Calibration

Coefficients for the supported archs are shipped in `coefs/<arch>.csv`. To re-fit
or adapt to a new arch, run `scripts/fit_autocp.sh`; the implementation lives in
`tools/autocp/`.

---

## 中文

**Auto CP** 用于决定*是否*开启卡内 CP，以及开启时*子序列长度*取多少。它用一个轻量的 kernel 时间模型预测若干候选配置的延迟，再把最优 CP 候选与不开 CP 作比较。

Auto CP 包含三个步骤：

1. **kernel 时间建模** —— 由 launch shape 预测每个 kernel 的耗时。
2. **CP 切分搜索** —— 只评估 wave 边界附近的少量子序列长度。
3. **CP 开启决策** —— 仅当最优 CP 候选以一定幅度优于不开 CP 时才开启。

### 1. kernel 时间建模

GDN kernel 的特点是都在重复某种固定粒度的工作单元（逐 chunk 递推或逐 CP 子序列修正）。我们用 CTA 承担的工作单元数抽象其负载，为每个 kernel row 拟合一条线性模型 $t_\kappa = a \cdot \mathrm{depth}_\kappa + b \cdot u + c$，其中：

- $\mathrm{depth}_\kappa$ —— **最大工作量**：通过贪心地把 CTA 分配到 SM 后，最忙 SM 上的工作单元数。由 `utils.py` 的 `schedule_depth()` 计算，模拟把 CTA runs 贪心打包到 $P$ 个 SM 上并返回耗时最长的 SM 的工作量。
- $u = H \cdot n_{\mathrm{part}}$ —— **CTA 数量**（总 CTA 数），刻画 CTA 增多带来的 launch/调度开销。这里不再除以 SM 数 $P$，$P$ 被吸收进拟合系数 $b$。
- $a, b, c$ —— 按 kernel 和硬件 arch 分别拟合。

在完成对每个 kernel 的建模后，FlashQLA Auto CP 能够接受 `infer` 和 `train` 两种模式，通过 `is_train` 控制。`infer` 模式下优化前向过程所有与 CP 相关 kernel 时间的总和，而`train` 模式则优化前向和反向过程中所有与 CP 相关的 kernel 时间的总和，分别用于推理和训练过程。

拟合系数存储在 `coefs/<arch>.csv` 中，每个 kernel 关于 `block_DV` 系数的映射关系由 `launch.py` 解决。

### 2. CP 切分搜索

候选子序列长度来自 `wave_candidates()`。对于含 $B$ 条序列、长度分别为 $S_b$ 的一个 batch，设有 $H$ 个 head、$P$ 个 SM，则子序列长度 $L$ 对应 $N_{\mathrm{cp}}(L) = \sum_b \lceil S_b / L \rceil$ 个 CP 分区，以及约 $w(L) = \lceil H \cdot N_{\mathrm{cp}} / P \rceil$ 个执行 wave。延迟在 $w(L)$ 跨越整数边界处变化最剧烈，因此 Auto CP 求解每个 wave 边界对应的 $L$（等长序列 $S$ 时，产生 $q$ 个 wave 的 $L \approx H B S / (q P)$），并按照 $L$ 降序搜索 `MAX_EVALS` 个候选得到最优 CP 切分。

### 3. CP 开启决策

由于开启 CP 会引入 overhead，因而在并行度充足的情况下不需要开启 CP。Auto CP 通过比较最优 CP 切分下的 CP 时间和关闭 CP 的时间来得到是否开启 CP 的决策。`use_cp` 当且仅当 $T_{\mathrm{cp}} < T_{\mathrm{nocp}} \cdot (1 - \mathrm{margin})$（margin 默认 0.05）。

### 用法

```python
from flash_qla.ops.gated_delta_rule.chunk.cp.autocp import decide

use_cp, lcp = decide(
    num_chunks=[...],      # 或 cu_seqlens=... / seq_lens=...
    num_v_heads=H,
    P=sm_count,            # 缺省取 CSV 中的 P
    chunk=chunk_size,      # 缺省取 CSV 中的 chunk
    is_train=False,
    g=gate_tensor,         # 可选；启用 warmup 感知的 prepare_h 深度
)
# use_cp: bool，lcp: max local chunks（不开 CP 时为 None）
```

已接入 FlashQLA 的 intracard CP 作为默认切分方式。

### 标定系数

支持架构的拟合系数已保存在 `coefs/<arch>.csv` 中。如果需要重新拟合/适应新架构，请使用 `scripts/fit_autocp.sh` 脚本进行拟合。相关实现位于 `tools/autocp/`