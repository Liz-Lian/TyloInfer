"""Build the portable CPU row fitter; no GPU architecture is required."""

import sys

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "tyloinfer._gyro_fit",
            sources=["tyloinfer/_gyro/module.c", "tyloinfer/_gyro/codebook.c"],
            depends=["tyloinfer/_gyro/fit_core.inc", "tyloinfer/_gyro/gyro_internal.h"],
            define_macros=[],
            libraries=[] if sys.platform == "win32" else ["m"],
        )
    ]
)
