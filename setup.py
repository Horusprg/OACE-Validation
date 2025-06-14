from setuptools import setup, find_packages

setup(
    name="oace-validation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "torchvision"
    ],
    author="Lyanh",
    description="OACE Validation - Otimização de Arquiteturas de Redes Neurais",
    python_requires=">=3.8",
) 