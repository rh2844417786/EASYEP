# Vendored fast-hadamard-transform

This directory contains the build-critical files from
`Dao-AILab/fast-hadamard-transform` tag `v1.1.0`, commit
`1cc807efbd6cc001df359822d60bf6052dd66859`.

Upstream: https://github.com/Dao-AILab/fast-hadamard-transform

License: BSD-3-Clause. The upstream `LICENSE` and `AUTHORS` files are preserved
in this directory. EASYEP applies two compatibility patches: the binding uses
PyTorch's lightweight `c10/cuda` stream and device-guard headers instead of the
cuSPARSE-pulling `ATen/cuda/CUDAContext.h`, and CUDA architecture generation is
delegated to `TORCH_CUDA_ARCH_LIST` (set to H100 `9.0` by the repair wrapper).
Trailing whitespace and final newlines were also normalized.

The files are vendored because the PyPI 1.1.0 source distribution omits the
`csrc/` inputs required by `setup.py`, while its CUDA-13 wheel selection attempts
an unrelated `cu122` release asset. EASYEP installs this directory with
`--offline --no-build-isolation --no-deps` and forces a local CUDA build.
