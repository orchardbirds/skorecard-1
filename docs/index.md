# skorecard

Tools for automating the construction of credit risk acceptance models (scorecards) in python, with a sklearn-compatible API.

Scorecard models are binary classification models that output the probability of default of a customer or customer's application, using Logistic Regression as the classification algorithm.

!!! warning
    This package is still under active development, and such the API is subject to major changes.

## Features ⭐

- Automate bucketing of features
- Dash webapp to help manually tweak bucketing of features in concordance with business knowledge
- Extension to `sklearn.linear_model.LogisticRegression` that is also able to report p-values
- Plots and reports to speed up the analysis and the writing of technical documentation.

## Installation

```shell
pip3 install git+ssh://git@gitlab.ing.net:2222/RiskandPricingAdvancedAnalytics/skorecard.git
```