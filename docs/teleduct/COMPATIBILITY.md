# Tested compatibility matrix (not yet filled)

Populate this table from *real measurements* on the target machine before claiming live support.

| OS/kernel | NVIDIA driver | GPU | CUDA toolkit/driver | Python | Engine + version | Model + precision | NVML | CUPTI | Status |
|---|---|---|---|---|---|---|---|---|---|
| Nobara Linux 44 | pending | RTX 3060 Mobile 6GB | pending | pending | pending | small model: pending | pending | pending | NOT YET TESTED |

Use lightweight commands first: `uname -r`, `nvidia-smi`, `python3 --version`, `git --version`.
`nvcc` availability does not imply CUPTI activity works with the serving process.
No automatic OS updates, driver reinstall or CUDA toolkit change are part of this project kit.
