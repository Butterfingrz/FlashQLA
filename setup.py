# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import os
import subprocess
from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.abspath(__file__))

rev = os.getenv("QLA_VERSION_SUFFIX", "")
if not rev:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        rev = "+" + subprocess.check_output(cmd, cwd=this_dir).decode("ascii").rstrip()
    except Exception:
        rev = ""

setup(
    name="flash_qla",
    version="0.1.2" + rev,
    description="FlashQLA: Fused TileLang kernels for Linear Attention",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    # tools/ is offline tooling (autocp calibration), never imported at run time.
    # It has no __init__.py anywhere, so find_packages() already skips it; the
    # exclude is there to say so out loud rather than to fix anything.
    packages=find_packages(exclude=["tools", "tools.*"]),
    # autocp reads its fitted coefficients from a CSV at run time -- there is no
    # in-code copy, so the wheel has to carry them. The pattern is relative to
    # each package dir and must name the subdirectory explicitly: coefs/ holds no
    # __init__.py, so find_packages() does not see it and a bare "*.csv" would
    # silently ship nothing. The coefficients are the only autocp data the wheel
    # needs; the shape sets that produced them live in tools/autocp/cases/.
    package_data={"": ["coefs/*.csv"]},
    license="MIT",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.8",
        "tilelang==0.1.9",
        "apache-tvm-ffi==0.1.9",
    ],
    zip_safe=False,
)
