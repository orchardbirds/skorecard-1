from setuptools import setup, find_packages


setup(
    name="skorecard",
    version="0.1",
    description="sklearn for Automated Credit Risk Modeling, 2020",
    author="RPAA",
    author_email="sandro.bjelogrlic@ing.com, ryan.chaves@ing.com, floriana.zefi@ing.com, daniel.timbrell@ing.com",
    license="ING Open Source",
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    url="https://gitlab.ing.net/RiskandPricingAdvancedAnalytics/skorecard",
    packages=find_packages(".", exclude=["tests", "notebooks"]),
)
