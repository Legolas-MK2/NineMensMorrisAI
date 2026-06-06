"""Build script for the fastnmm C++ extension."""
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

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
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
