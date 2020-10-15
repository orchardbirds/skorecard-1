# Contributing guidelines

## Workflow

Get an overview of on-going and upcoming work on the [skorecard GitLab Board.](https://gitlab.ing.net/RiskandPricingAdvancedAnalytics/skorecard/boards)

Our workflow is the following:

- Start on the skorecard GitLab with an Issue. In the Issue, click "Create merge request". Then in this created MR,
 click "Check out branch".
 
- In a local terminal, check out branch with `git`.

- Code. Add files. Commit files. Then `git pull origin master`. Resolve conflicts if any (then add/commit). Push.

- Ask a colleague to review the MR on GitLab (e.g. by tagging them or assigning them). Discuss and resolve. Add additional code
and commits if necessary.

- Ask a colleague to approve the MR on GitLab.

- Merge to master on GitLab.

## Repo structure

``` nohighlight
├── notebooks                  <- Jupyter notebooks are only allowed here (will be moved to docs/ folder instead!)
├── skorecard                  <- All reusable source code
├── docs                       <- Documentation in markdown
└── tests                      <- Tests for unit and functional testing
.gitignore                     <- Files for git to ignore
.gitlab-ci.yml                 <- Setup file for CI pipeline
.pre-commit-config.yaml        <- Setup file for pre-commit pipeline
LICENSE                        <- License for potential open-sourcing
MANIFEST.in                    <- Which non-python files to include with the built package
mkdocs.yml                     <- Configuration file for building MkDocs documentation website
README.md                      <- General information about the project and repo structure, how to use it, etc.
requirements.txt               <- Python dependencies (packages & versions)
setup.py                       <- To setup skorecard package
```

## Coding standards

Please review [RPAA Coding Standards](https://confluence.europe.intranet/display/RPAT/RPAA+Coding+Standards) and
[RPAA Best Practices](https://confluence.europe.intranet/display/RPAT/RPAA+Best+Practices).

## Development setup

We use python editable installs to develop this package. We added all packages also needed for development to the "all" optional dependency set. To install:

```shell
git clone git+ssh://git@gitlab.ing.net:2222/RiskandPricingAdvancedAnalytics/skorecard.git
cd skorecard
pip3 install -e ".[all]"
```

You can run the unittests with:

```bash
pytest
```

We use [pre-commit](https://pre-commit.com/) to ensure code quality. To set it up:

```bash
pre-commit install
```

We use [SemVer](http://semver.org/) for versioning.

We use [mkdocs](https://www.mkdocs.org/) for documentation. You can view the docs locally with:

```bash
mkdocs serve
```

## Terminology

- `BucketMapping` is a custom class that stores all the information needed for bucketing, including the map itself (either boundaries for binning, or a list of lists for categoricals)
- `FeaturesBucketMapping` is simply a collection of `BucketMapping`s, and is used to store all info for bucketing transformations for a dataset.

## README badges

Because this package has not been released on pypi yet, we cannot use shields.io. 
We used [pybadges](https://github.com/google/pybadges) to generate some of the badges:

```bash
python -m pybadges \
    --left-text="python" \
    --right-text="3.6+" \
    --whole-link="https://www.python.org/" > docs/assets/img/python_badge.svg
```
