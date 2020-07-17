from setuptools import setup, find_packages

setup(
    name="auto-cr",
    version="0.1",
    description="Automated Credit Risk Modeling, 2020",
    author="RPAA",
    author_email="sandro.bjelogrlic@ing.com, ryan.chaves@ing.com, floriana.zefi@ing.com",
    packages=find_packages(".", exclude=["tests", "notebooks"]),
)
