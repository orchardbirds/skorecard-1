# skorecard

Tools for automating building credit risk acceptance models (scorecards) in python, with a sklearn-compatible API.

Scorecard models are binary classification models that output the probability of default of a customer or customer's application, and Logistic Regression as the classification algorithm.

!!! warning
    This package is still under active development, and such the API is subject to major change.

## Features ⭐

- Automate bucketing of features
- Dash webapp to help manually tweak bucketing of features with business knowledge
- Extension to `sklearn.linear_model.LogisticRegression` that is also able to report p-values
- Plots and reports to speed up analysis and writing technical documentation.

## Installation

```shell
pip3 install git+ssh://git@gitlab.ing.net:2222/RiskandPricingAdvancedAnalytics/skorecard.git
```
