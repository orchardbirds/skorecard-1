# Contributing guidelines

## Workflow

Get an overview of on-going and upcoming work on the skorecard GitLab Board.

Our work flow is the following:
* Start on the skorecard GitLab with an Issue. In the Issue, "Create merge request". Then in MR, "Check out branch".
* In a local terminal, check out branch with `git`.
* Code. Add files. Commit files. Then `git pull origin master`. Resolve conflicts if any (then add/commit). Push.
* Ask colleague to review MR on GitLab (e.g. by tagging them or assigning them). Discuss and resolve. Additional code
and commits as necessary.
* Ask colleague to Approve MR on GitLab.
* Merge to master on GitLab.

## Repo structure

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

## Coding standards

Please review [RPAA Coding Standards](https://confluence.europe.intranet/display/RPAT/RPAA+Coding+Standards) and
[RPAA Best Practices](https://confluence.europe.intranet/display/RPAT/RPAA+Best+Practices).

## Development setup

We use python editable installs to develop this package:

```shell
git clone git+ssh://git@gitlab.ing.net:2222/RiskandPricingAdvancedAnalytics/skorecard.git
cd skorecard
pip3 install -e .
```

You'll also need to install packages required for development:

```bash
pip install -r requirements.txt
pip install -r tests/test_requirements.txt
```

You can run the unittests with:

```bash
pytest
```

We use [pre-commit](https://pre-commit.com/) to ensure code quality. To set it up:

```bash
pip install pre-commit
pre-commit install
```

We use [SemVer](http://semver.org/) for versioning.

We use [mkdocs](https://www.mkdocs.org/) for documentation, you can view the docs locally with:

```bash
mkdocs serve
```

## Terminology

- `features_bucket_mapping` Is a `dict`-like object, containing all the features and the info for bucketing

```yml
{
    'type' : 'categorical', # or numerical
    'missing_bucket' : None, # error, or bucket index number
    'boundaries' : [...] # or None if categorical
    'map' : [ ['a','b'], ['c']] # or None if numerical
}
```


## README badges

Because this package has not been released on pypi yet, we cannot use shields.io. 
We used [pybadges]() to generate some of the badges:

```bash
python -m pybadges \
    --left-text="python" \
    --right-text="3.6+" \
    --whole-link="https://www.python.org/" > docs/assets/img/python_badge.svg
```
