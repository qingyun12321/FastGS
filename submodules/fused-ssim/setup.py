import os


def _ensure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    print("CUDA_HOME is not set; defaulting to /usr/local/cuda")


_ensure_cuda_home()

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _ensure_torch_cuda_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    # Fallback for container builds where no GPU is visible during compilation.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6+PTX"
    print(
        "TORCH_CUDA_ARCH_LIST is not set; defaulting to "
        f"{os.environ['TORCH_CUDA_ARCH_LIST']}"
    )


_ensure_torch_cuda_arch_list()

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=[
            "ssim.cu",
            "ext.cpp"])
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
