import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = 'Debug' if self.debug else 'Release'

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
            '-DBUILD_PYTHON_BINDINGS=ON',
            '-DBUILD_LIB=OFF',  # Don't build library and CLI tools
        ]

        build_args = ['--config', cfg]

        # Platform-specific settings
        if sys.platform.startswith('darwin'):
            # Metal is default on macOS
            cmake_args += ['-DUSE_METAL=ON']
        
        if sys.platform.startswith('linux'):
            # Optionally enable CUDA on Linux
            cmake_args += ['-DBUILD_CUDA=OFF']

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if 'CMAKE_BUILD_PARALLEL_LEVEL' not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, 'parallel') and self.parallel:
                build_args += [f'-j{self.parallel}']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


# Read the README for long description
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='knapsack-solver',
    version='2.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Fast knapsack solver with beam search and scout mode for exact solver integration',
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=[CMakeExtension('knapsack_py')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[],
    extras_require={
        'dev': ['pytest', 'numpy'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
    ],
    keywords='knapsack optimization beam-search solver operations-research',
    project_urls={
        'Source': 'https://github.com/yourusername/knapsack',
        'Bug Reports': 'https://github.com/yourusername/knapsack/issues',
    },
)
