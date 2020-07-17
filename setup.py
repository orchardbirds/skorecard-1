import os

from setuptools import setup, find_packages


def read(fname):
    """Helper function to read filename from repo path."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="auto-cr",
    version="0.1",
    description="Automated Credit Risk Modeling, 2020",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="RPAA",
    author_email="sandro.bjelogrlic@ing.com, ryan.chaves@ing.com, floriana.zefi@ing.com",
    license="ING Open Source",
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://gitlab.com/ing_rpaa/probatus",
    packages=find_packages(".", exclude=["tests", "notebooks"]),
)
