from setuptools import setup, find_packages

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

base_packages = [
    "scipy>=1.5.2",
    "numpy>=1.19.2",
    "pandas>=1.1.2",
    "scikit-learn>=0.23.2",
    "probatus>=1.1.1",
]

dashboard_dep = [
    "dash>=1.15.0",
    "jupyter-dash>=0.3.0",
    "dash_bootstrap_components>=0.10.6",
]

reporting_dep = ["plotly>=4.5.1"]

dev_dep = [
    "flake8>=3.8.3",
    "black>=19.10b0",
    "pre-commit>=2.5.0",
    "mypy>=0.770",
    "flake8-docstrings>=1.4.0" "pytest>=6.0.0",
    "pytest-cov>=2.10.0",
]


docs_dep = [
    "mkdocs>=1.1",
    "mkdocs-material>=5.5.12",
    "mkdocstrings>=0.13.2",
    "mkdocs-git-revision-date-localized-plugin>=0.7.2",
]

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
    install_requires=base_packages,
    extras_require={
        "base": base_packages,
        "dashboard": dashboard_dep,
        "reporting": reporting_dep,
        "all": base_packages + dashboard_dep + reporting_dep + dev_dep + docs_dep,
    },
    url="https://gitlab.ing.net/RiskandPricingAdvancedAnalytics/skorecard",
    packages=find_packages(".", exclude=["tests", "notebooks", "docs"]),
)
