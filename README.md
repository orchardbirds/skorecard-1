# skorecard

[![pipeline status](https://gitlab.ing.net/RiskandPricingAdvancedAnalytics/skorecard/badges/master/pipeline.svg)](https://gitlab.ing.net/RiskandPricingAdvancedAnalytics/skorecard/commits/master)
[![coverage report](https://gitlab.ing.net/RiskandPricingAdvancedAnalytics/skorecard/badges/master/coverage.svg)](https://gitlab.ing.net/RiskandPricingAdvancedAnalytics/skorecard/commits/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**Automate building of credit risk (CR) models**


## Repo/project description

This codebase aims to provide modular components to automate various common steps encountered during the building of a 
CR models, with particular attention to "scorecard" credit decision models (CDMs). In general, CDMs are binary 
classification models that output the probability of default of a customer or customer's application. "Scorecard" here
refers to a CDM that uses Logistic Regression as the classification algorithm.


## Repository structure

``` nohighlight
├── data                       <- Data sources (.gitignored)
├── notebooks                  <- Jupyter notebooks are only allowed here
│   ├── private                <- Private notebooks (.gitignored)
│   ├── review                 <- Notebooks temporarily here for others to review, e.g. related to a MR
│   └── demo                   <- Notebooks to demonstrate a function, analysis, modeling, etc. for posterity
├── skorecard                  <- All reusable source code
└── tests                      <- Tests for unit and functional testing
   
.gitignore                     <- Files for git to ignore
.gitlab-ci.yml                 <- Setup file for CI pipeline
.pre-commit-config.yaml        <- Setup file for pre-commit pipeline
LICENSE                        <- License for potential open-sourcing
main.py                        <- Main skorecard entry point
README.md                      <- General information about the project and repo structure, how to use it, etc.
requirements.txt               <- Python dependencies (packages & versions)
setup.py                       <- To setup skorecard package locally via pip
```


## Goal

Our aim is to save data scientists' time during model development by automating a number of common steps, such as the
bucketing of features in scorecard CDMs.


## People

| Role              | Person              | Team           |
| :---------------- | :------------------ | :------------- |
| Data Scientist    | Sandro Bjelogrlic   | RPAA           |
| Data Scientist    | Ryan Chaves         | RPAA           |
| Data Scientist    | Floriana Zefi       | RPAA           |
| Data Scientist    | Daniel Timbrell     | RPAA           |
| Code Reviewer     | TBD                 | RPAA           |


## Methods

We create custom sklearn [Transformers](https://scikit-learn.org/stable/data_transforms.html) for easy integration into
sklearn Pipelines. We use YAML files to store configuration information in a way this is editable by both the developer
manually and automatically by the Transformers.


## How to run

* `$ pip3 install -e .` to setup package locally
* `$ python3 main.py` to run a sample skorecard pipeline


## Contributing

Get an overview of on-going and upcoming work on the skorecard GitLab Board.

Our work flow is the following:
* Start on the skorecard GitLab with an Issue. In the Issue, "Create merge request". Then in MR, "Check out branch".
* In a local terminal, check out branch with `git`.
* Code. Add files. Commit files. Then `git pull origin master`. Resolve conflicts if any (then add/commit). Push.
* Ask colleague to review MR on GitLab (e.g. by tagging them or assigning them). Discuss and resolve. Additional code
and commits as necessary.
* Ask colleague to Approve MR on GitLab.
* Merge to master on GitLab.

Please review [RPAA Coding Standards](https://confluence.europe.intranet/display/RPAT/RPAA+Coding+Standards) and
[RPAA Best Practices](https://confluence.europe.intranet/display/RPAT/RPAA+Best+Practices).


## Requirements

Developed with Python 3.6 (for compatibility with python version currently on GitLab Runner)

Versions of packages are in `requirements.txt`.
