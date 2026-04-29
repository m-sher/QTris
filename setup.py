from setuptools import setup, find_packages, Extension

pathfinder_module = Extension(
    "TetrisEnv.pathfinder",
    sources=["TetrisEnv/pathfinder.c"],
    extra_compile_args=["-O3", "-std=c99"],
)

hole_finder_module = Extension(
    "TetrisEnv.hole_finder",
    sources=["TetrisEnv/hole_finder.c"],
    extra_compile_args=["-O3", "-std=c99"],
)

b2b_search_module = Extension(
    "TetrisEnv.b2b_search",
    sources=["TetrisEnv/b2b_search.c"],
    extra_compile_args=["-O3", "-std=c99", "-lm"],
)

setup(
    name="TetrisEnv",
    version="0.8.0",
    description="Environment for RL with Tetris SRS+",
    author="Michael Sherrick",
    author_email="michael.a.sherrick@gmail.com",
    url="https://github.com/m-sher/TFTetrisEnv",
    packages=find_packages(),
    ext_modules=[pathfinder_module, hole_finder_module, b2b_search_module],
    install_requires=[
        "tensorflow",
        "tf-agents",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
