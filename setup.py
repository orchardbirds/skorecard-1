from setuptools import setup, find_packages


setup(
    name="autoCR",
    version="0.1",
    description="Automated Credit Risk Modeling, 2020",
    author="RPAA",
    author_email="sandro.bjelogrlic@ing.com, ryan.chaves@ing.com, floriana.zefi@ing.com",
    license="ING Open Source",
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://gitlab.ing.net/RiskandPricingAdvancedAnalytics/autoCR",
    packages=find_packages(".", exclude=["tests", "notebooks"]),
)
