#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

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
    name="diff_gaussian_rasterization_fastgs",
    packages=['diff_gaussian_rasterization_fastgs'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_fastgs._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "cuda_rasterizer/adam.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
