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

from setuptools import find_packages, setup
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

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="simple_knn",
    packages=find_packages(include=["simple_knn", "simple_knn.*"]),
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)
