"""Build script for the fastnmm C++ extension."""
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# librt provides shm_open / shm_unlink on Linux (glibc < 2.34).
# It's a no-op on macOS and irrelevant on Windows.
extra_libs = []
if sys.platform.startswith("linux"):
    extra_libs.append("rt")

ext_modules = [
    Pybind11Extension(
        "fastnmm._core",
        sources=[
            "src/fastnmm/_core/nine_mens_morris.cpp",
            "src/fastnmm/_core/bindings.cpp",
        ],
        include_dirs=["src/fastnmm/_core"],
        cxx_std=17,
        extra_compile_args=["-O3", "-funroll-loops", "-fvisibility=hidden"],
        libraries=extra_libs,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
