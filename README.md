# skorecard

![pytest](https://github.com/timvink/skorecard/workflows/Release/badge.svg)
![pytest](https://github.com/timvink/skorecard/workflows/Development/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Tools for automating building credit risk acceptance models (scorecards) in python, with a sklearn-compatible API.

Scorecard models are binary classification models that output the probability of default of a customer or customer's application, and Logistic Regression as the classification algorithm.

## Features ⭐

- Automate bucketing of features
- Dash webapp to help manually tweak bucketing of features with business knowledge
- Extension to `sklearn.linear_model.LogisticRegression` that is also able to report p-values
- Plots and reports to speed up analysis and writing technical documentation.

## Installation

```shell
pip3 install git+ssh://git@gitlab.ing.net:2222/RiskandPricingAdvancedAnalytics/skorecard.git
```

## Documentation

*Documentation website yet to be built*
