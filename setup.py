# setup.py
from setuptools import setup, find_packages

setup(
    name="oned_Vlasov_Poisson_jax_solver",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "diffrax",
        "numpy",
        "matplotlib",
        "scipy",
        "pandas"
    ],
)
