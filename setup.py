# setup.py
from setuptools import setup, find_packages

setup(
    name="Oned_Vlasov_Poisson_jax_solver",
    version="0.1.0",
    author="Jianfeng Ye",
    author_email="xyjjeff23@gmail.com",
    description="A fast 1D Vlasov solver using JAX",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/XYJeff23/1D_Vlasov_JAX_Solver",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[  # 指定依赖项
        "jax",
        "jaxlib",
        "diffrax",
        "numpy",
        "matplotlib",
        "scipy",
        "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
