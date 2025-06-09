from setuptools import setup, find_packages

setup(
    name="dna-brush-paper-2025",
    version="0.1.0",
    description="All the code for DNA Brush Paper 2025 in one place",
    author="Jonathan Bostock",
    author_email="jdb22@ic.ac.uk",
    packages=find_packages(),
    install_requires=[
        "scipy",
        "numpy", 
        "matplotlib",
        "seaborn"
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
