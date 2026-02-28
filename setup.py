from setuptools import setup, find_packages

setup(
    name="docqa",
    version="0.1.0",
    description="RAG-powered document Q&A from the command line",
    author="Joerg Michno",
    author_email="michno.jrg@gmail.com",
    url="https://github.com/joergmichno/docqa",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "llm": ["anthropic>=0.18.0"],
        "dev": ["pytest>=7.0", "pytest-cov>=4.0"],
    },
    entry_points={
        "console_scripts": [
            "docqa=docqa.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
