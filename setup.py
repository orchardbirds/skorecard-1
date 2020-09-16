from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="skorecard",
    version="0.1",
    description="Tools for building scorecard models in python, with a sklearn-compatible API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ING, Risk and Pricing Advanced Analytics",
    author_email="ML_Risk_and_Pricing_AA@ing.com",
    license="ING Open Source",
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    include_package_data=True,
    install_requires=["scipy>=1.5.2", "numpy>=1.19.2", "pandas>=1.1.2", "scikit-learn>=0.23.2", "probatus>=1.1.1"],
    url="https://gitlab.ing.net/RiskandPricingAdvancedAnalytics/skorecard",
    packages=find_packages(".", exclude=["tests", "notebooks", "docs"]),
)
