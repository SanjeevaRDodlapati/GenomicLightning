"""Setup configuration for GenomicLightning."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the version from the package
def get_version():
    """Get version from VERSION file or pyproject.toml."""
    try:
        # Try VERSION file first
        with open("VERSION", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        pass
    
    # Fall back to pyproject.toml using regex (no external deps needed)
    try:
        import re
        with open("pyproject.toml", "r") as f:
            content = f.read()
            # Try modern [project] section first
            match = re.search(r'version\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
            # Try legacy [tool.poetry] section
            match = re.search(r'\[tool\.poetry\].*?version\s*=\s*"([^"]+)"', content, re.DOTALL)
            if match:
                return match.group(1)
    except (FileNotFoundError, ImportError):
        pass
    
    # Try importing from package if already installed
    try:
        import genomic_lightning
        return getattr(genomic_lightning, '__version__', "0.1.0")
    except ImportError:
        pass
    
    # Final fallback
    return "0.1.0"

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
try:
    with open("requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    # Fallback requirements if requirements.txt doesn't exist
    requirements = [
        "torch>=1.8.0",
        "pytorch-lightning>=1.5.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.50.0",
        "pyyaml>=5.4.0",
        "h5py>=3.1.0",
        "tensorboard>=2.4.0",
    ]

setup(
    name="genomic-lightning",
    version=get_version(),
    author="Sean Doolan",
    author_email="sdodl001@odu.edu",
    description="PyTorch Lightning framework for genomic deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sdodla/GenomicLightning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.10",
        ],
        "docs": [
            "sphinx>=3.5",
            "sphinx-rtd-theme>=0.5",
            "sphinx-autodoc-typehints>=1.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "genomic-lightning=genomic_lightning.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
