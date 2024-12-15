from setuptools import setup, find_packages
from pathlib import Path

# Read README content
this_directory = Path(__file__).parent
long_description = (this_directory / "docs/README-CLI.md").read_text()
requirements = (this_directory / "requirements-cli.txt").read_text().splitlines()

setup(
    name="felafax-cli",
    version="0.1.17",
    description="CLI tool for interacting with the Felafax platform; fine-tuning and inference on non-NVIDIA chipsets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nikhil Sonti",
    author_email="nikhil@felafax.ai",
    packages=find_packages(),
    package_dir={},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "felafax-cli=felafax.cli.main:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)

