# Contributing guidelines

Make sure to discuss any changes you would like to make in the issue board, before putting in any work.

## Setup

Development install:

```shell
pip install -e '.[all]'
```

Unit testing:

```shell
pytest
```

We use [pre-commit](https://pre-commit.com/) hooks to ensure code styling. Install with:

```shell
pre-commit install
```

## Documentation

We use [mkdocs](https://www.mkdocs.org) with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) theme. The docs are structured using the [divio documentation system](https://documentation.divio.com/). To view the docs locally:

```shell
pip install mkdocs-material
mkdocs serve
```


## Terminology

- `BucketMapping` is a custom class that stores all the information needed for bucketing, including the map itself (either boundaries for binning, or a list of lists for categoricals)
- `FeaturesBucketMapping` is simply a collection of `BucketMapping`s, and is used to store all info for bucketing transformations for a dataset.

