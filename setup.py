from pathlib import Path

from setuptools import find_packages, setup

NAME = "mapa"
VERSION = "0.1.0"
DESCRIPTION = "A masked autoencoder for intracranial EEG."
LICENSE = "Apache-2.0"


def get_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",
        url="https://github.com/bentang18/MAPA",
        author="Ben Tang, Zachary Spalding, Gregory B. Cogan",
        author_email="gregory.cogan@duke.edu",
        packages=find_packages(include=["mapa", "mapa.*"]),
        license=LICENSE,
        license_files=("LICENSE", "NOTICE"),
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
        python_requires=">=3.10",
        install_requires=get_requirements(),
        extras_require={"test": ["pytest>=8"], "preprocessing": ["mne>=1.6,<2"]},
        package_data={"mapa.preprocessing": ["*.json"]},
    )
