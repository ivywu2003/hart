import io
import os
import re
import subprocess
import warnings
from typing import List, Set

import setuptools
import torch
from packaging.version import Version, parse
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension, CppExtension

import pyopencl as cl

ROOT_DIR = os.path.dirname(__file__)

# Check if we're on macOS and can use Metal
IS_MACOS = (os.uname()[0] == 'Darwin')
available_backends = dir(torch.backends)

# Print the backends
print("Available backends:")
for backend in available_backends:
    print(backend)
MPS_ENABLED = IS_MACOS and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

def check_opencl():
    try:
        # Get all OpenCL platforms
        platforms = cl.get_platforms()
        
        # Check if any of the platforms are Intel-based
        for platform in platforms:
            devices = platform.get_devices()
            for device in devices:
                if 'Intel' in device.name:
                    print("Intel OpenCL platform is available.")
                    return True
        print("Intel OpenCL platform is not available.")
        return False
    except cl.LogicError:
        print("OpenCL is not available.")
        return False

XPU_ENABLED = IS_MACOS and check_opencl()


# Supported NVIDIA GPU architectures.
SUPPORTED_ARCHS = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}

# Compiler flags.
CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
if IS_MACOS:
    if MPS_ENABLED:
        CXX_FLAGS = ["-g", "-O3", "-std=c++17", "-DENABLE_BF16"]  # Remove OpenMP flags on macOS
    elif XPU_ENABLED:
        CXX_FLAGS = ["-g", "-O3", "-std=c++14", "-DENABLE_BF16"]  # Remove OpenMP flags on macOS

NVCC_FLAGS = [
    "-O2",
    "-std=c++17",
    "-DENABLE_BF16",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--threads=8",
]

if XPU_ENABLED: # change c++ version
    NVCC_FLAGS = [
    "-O2",
    "-std=c++14",
    "-DENABLE_BF16",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--threads=8",
]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

if IS_MACOS:
    if MPS_ENABLED:
        from distutils.unixccompiler import UnixCCompiler
        if '.mm' not in UnixCCompiler.src_extensions:
            UnixCCompiler.src_extensions.append('.mm')
            UnixCCompiler.language_map['.mm'] = 'objc'

        ext_modules = []
        fused_kernel_extension = CppExtension(
            name='hart_backend.fused_kernels',
            sources=[
                "metal/kernel_library.mm",
            ],
            extra_compile_args={
                "cxx": [
                    '-Wall', 
                    '-std=c++17',
                    '-framework', 
                    'Metal', 
                    '-framework', 
                    'Foundation',
                    '-ObjC++'
                ],
            },
        )
        ext_modules.append(fused_kernel_extension)
    elif XPU_ENABLED:
        from distutils.unixccompiler import UnixCCompiler
        if '.cl' not in UnixCCompiler.src_extensions:
            UnixCCompiler.src_extensions.append('.cl')

        ext_modules = []
        fused_kernel_extension = CppExtension(
            name='hart_backend.fused_kernels',
            sources=[
                "opencl/rope/fused_rope.cl",
                "opencl/rope/fused_rope_with_pos.cl",
                "opencl/layernorm/layernorm_kernels.cl",
                "opencl/layernorm/layernorm_kernels.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
            },
        )
        ext_modules.append(fused_kernel_extension)
    else:
        raise RuntimeError(
            "MPS/XPU is not available on macOS. MPS/XPU must be available to build the package."
        )

else:
    if CUDA_HOME is None:
        raise RuntimeError(
            "Cannot find CUDA_HOME. CUDA must be available to build the package."
        )

    def get_nvcc_cuda_version(cuda_dir: str) -> Version:
        """Get the CUDA version from nvcc.

        Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
        """
        nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], text=True)
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = parse(output[release_idx].split(",")[0])
        return nvcc_cuda_version

    def get_torch_arch_list() -> Set[str]:
        # TORCH_CUDA_ARCH_LIST can have one or more architectures,
        # e.g. "8.0" or "7.5,8.0,8.6+PTX". Here, the "8.6+PTX" option asks the
        # compiler to additionally include PTX code that can be runtime-compiled
        # and executed on the 8.6 or newer architectures. While the PTX code will
        # not give the best performance on the newer architectures, it provides
        # forward compatibility.
        env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if env_arch_list is None:
            return set()

        # List are separated by ; or space.
        torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
        if not torch_arch_list:
            return set()

        # Filter out the invalid architectures and print a warning.
        valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})
        arch_list = torch_arch_list.intersection(valid_archs)
        # If none of the specified architectures are valid, raise an error.
        if not arch_list:
            raise RuntimeError(
                "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
                f"variable ({env_arch_list}) is supported. "
                f"Supported CUDA architectures are: {valid_archs}."
            )
        invalid_arch_list = torch_arch_list - valid_archs
        if invalid_arch_list:
            warnings.warn(
                f"Unsupported CUDA architectures ({invalid_arch_list}) are "
                "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
                f"({env_arch_list}). Supported CUDA architectures are: "
                f"{valid_archs}."
            )
        return arch_list

    # First, check the TORCH_CUDA_ARCH_LIST environment variable.
    compute_capabilities = get_torch_arch_list()
    if not compute_capabilities:
        # If TORCH_CUDA_ARCH_LIST is not defined or empty, target all available
        # GPUs on the current machine.
        device_count = torch.cuda.device_count()
        gpu_version = None
        for i in range(device_count):
            major, minor = torch.cuda.get_device_capability(i)
            if gpu_version is None:
                gpu_version = f"{major}{minor}"
            else:
                if gpu_version != f"{major}{minor}":
                    raise RuntimeError(
                        "Kernels for GPUs with different compute capabilities cannot be installed simultaneously right now.\nPlease use CUDA_VISIBLE_DEVICES to specify the GPU for installation."
                    )
            if major < 7:
                raise RuntimeError(
                    "GPUs with compute capability below 7.0 are not supported."
                )
            compute_capabilities.add(f"{major}.{minor}")
    else:
        if len(compute_capabilities) > 1:
            raise RuntimeError(
                "Kernels for GPUs with different compute capabilities cannot be installed simultaneously right now.\nPlease restrict the length of TORCH_CUDA_ARCH_LIST to 1."
            )
        else:
            gpu_version = compute_capabilities[0].replace(".", "")

    nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
    if not compute_capabilities:
        # If no GPU is specified nor available, add all supported architectures
        # based on the NVCC CUDA version.
        compute_capabilities = SUPPORTED_ARCHS.copy()
        if nvcc_cuda_version < Version("11.1"):
            compute_capabilities.remove("8.6")
        if nvcc_cuda_version < Version("11.8"):
            compute_capabilities.remove("8.9")
            compute_capabilities.remove("9.0")

    # Validate the NVCC CUDA version.
    if nvcc_cuda_version < Version("11.0"):
        raise RuntimeError("CUDA 11.0 or higher is required to build the package.")
    if nvcc_cuda_version < Version("11.1"):
        if any(cc.startswith("8.6") for cc in compute_capabilities):
            raise RuntimeError(
                "CUDA 11.1 or higher is required for compute capability 8.6."
            )
    if nvcc_cuda_version < Version("11.8"):
        if any(cc.startswith("8.9") for cc in compute_capabilities):
            # CUDA 11.8 is required to generate the code targeting compute capability 8.9.
            # However, GPUs with compute capability 8.9 can also run the code generated by
            # the previous versions of CUDA 11 and targeting compute capability 8.0.
            # Therefore, if CUDA 11.8 is not available, we target compute capability 8.0
            # instead of 8.9.
            warnings.warn(
                "CUDA 11.8 or higher is required for compute capability 8.9. "
                "Targeting compute capability 8.0 instead."
            )
            compute_capabilities = {
                cc for cc in compute_capabilities if not cc.startswith("8.9")
            }
            compute_capabilities.add("8.0+PTX")
        if any(cc.startswith("9.0") for cc in compute_capabilities):
            raise RuntimeError(
                "CUDA 11.8 or higher is required for compute capability 9.0."
            )

    # Add target compute capabilities to NVCC flags.
    for capability in compute_capabilities:
        num = capability[0] + capability[2]
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if capability.endswith("+PTX"):
            NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

    # Use NVCC threads to parallelize the build.
    if nvcc_cuda_version >= Version("11.2"):
        num_threads = min(os.cpu_count(), 8)
        NVCC_FLAGS += ["--threads", str(num_threads)]

    ext_modules = []


    # rope from transformer engine
    fused_kernel_extension = CUDAExtension(
        name="hart_backend.fused_kernels",
        sources=[
            "csrc/rope/fused_rope.cu",
            "csrc/rope/fused_rope_with_pos.cu",
            "csrc/layernorm/layernorm_kernels.cu",
            "csrc/pybind.cpp",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(fused_kernel_extension)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str):
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return open(get_path("README.md"), encoding="utf-8").read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


def load_kernel(file_path, context):
    with open(file_path, 'r') as kernel_file:
        kernel_source = kernel_file.read()
    program = cl.Program(context, kernel_source).build(options=["-cl-std=CL1.2"])
    return program

if XPU_ENABLED:
    platforms = cl.get_platforms()
    platform = platforms[0]  # Use the first platform
    devices = platform.get_devices()
    context = cl.Context(devices)

    load_kernel("opencl/rope/fused_rope.cl", context)
    load_kernel("opencl/rope/fused_rope_with_pos.cl", context)
    load_kernel("opencl/layernorm/layernorm_kernels.cl", context)

else:
    setuptools.setup(
        name="hart_backend",
        version="0.1.0",
        author="HART team, MIT HAN Lab",
        license="Apache 2.0",
        description=(
            "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"
        ),
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3.11",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        packages=setuptools.find_packages(
            exclude=("benchmarks", "csrc", "docs", "examples", "tests")
        ),
        python_requires=">=3.8",
        # install_requires=get_requirements(),
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
    )



