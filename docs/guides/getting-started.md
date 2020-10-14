# Getting started guide

A large part of developing scorecards is bucketing the features. `skorecard` provides a range of different [bucketers](/api/bucketers.html) to use.
For example the [AgglomerativeClusteringBucketer](/api/bucketers.html#skorecard.bucketers.bucketers.AgglomerativeClusteringBucketer):

```python
from skorecard import datasets
from skorecard.bucketers import AgglomerativeClusteringBucketer

X, y = datasets.load_uci_credit_card(return_X_y=True)
bucketer = AgglomerativeClusteringBucketer(bins = 10, variables=['LIMIT_BAL'])
bucketer.fit_transform(X)
```

## Scikit-learn pipeline

These bucketers are [scikit-learn](http://scikit-learn.org/) compatible, which means you can use them inside a pipeline:

=== "Basic sklearn pipeline"

    ```python
    from skorecard import datasets
    from skorecard.bucketers import EqualWidthBucketer, OrdinalCategoricalBucketer

    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    X, y = datasets.load_uci_credit_card(return_X_y=True)

    bucket_pipeline = make_pipeline(
        EqualWidthBucketer(bins=5, variables=['LIMIT_BAL', 'BILL_AMT1']),
        OrdinalCategoricalBucketer(variables=['EDUCATION', 'MARRIAGE'])
    )

    pipeline = Pipeline([
        ('bucketing', bucket_pipeline),
        ('one-hot-encoding', OneHotEncoder()),
        ('lr', LogisticRegression())
    ])

    pipeline.fit(X, y)
    f"AUC = {roc_auc_score(y, pipeline.predict_proba(X)[:,1]):.4f}"
    ```

=== "With ColumnTransformer"

    ```python
    from skorecard import datasets
    from skorecard.bucketers import EqualWidthBucketer, OrdinalCategoricalBucketer

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    bucket_pipeline = ColumnTransformer([
        ('categorical_preprocessing', OrdinalCategoricalBucketer(), ['EDUCATION', 'MARRIAGE']),
        ('numerical_preprocessing', EqualWidthBucketer(bins=5), ['LIMIT_BAL','BILL_AMT1'])
    ], remainder="passthrough")

    pipeline = Pipeline([
        ('bucketing', bucket_pipeline),
        ('one-hot-encoding', OneHotEncoder()),
        ('lr', LogisticRegression())
    ])

    pipeline.fit(X, y)
    f"AUC = {roc_auc_score(y, pipeline.predict_proba(X)[:,1]):.4f}"
    ```

=== "With FeatureUnion"

    ```python
    # TODO!
    ```


### Bucketing categorical vs numerical features

`skorecard   offers different bucketers for numerical and categorical features.
TODO: link to api.

You can use a util function to attempt to auto-detect column types:

```python
from skorecard import datasets
from skorecard.bucketers import EqualWidthBucketer, OrdinalCategoricalBucketer
from skorecard.utils import detect_types
from sklearn.pipeline import make_pipeline

X, y = datasets.load_uci_credit_card(return_X_y=True)

num_cols, cat_cols = detect_types(X)

bucket_pipeline = make_pipeline(
    EqualWidthBucketer(bins=5, variables=num_cols),
    OrdinalCategoricalBucketer(variables=cat_cols)
)
bucket_pipeline.fit_transform(X)
```

### Making manual changes to the buckets

You'll often want to incorporate business logic as well. To change the buckets, you'll first need to extract the values from the pipeline. `skorecard` offers the [get_features_bucket_mapping](api/pipeline.md) function to do this:

```python
from skorecard.pipeline import get_features_bucket_mapping

features_bucket_mapping = get_features_bucket_mapping(pipe)
```

Next up, you'll want to start a webapp to make changes interactively.
Recommended to this inside a notebooks.

```python
app = ManualBucketerApp(X, features_bucket_mapping)
app.run_server(mode="external")
app.stop_server()

# Access the updated features bucket mapping
app.features_bucket_mapping
```

### Using bucket mapping 

Once you have boundaries you like, you save save them as a yml and use them later

```python
# TODO: demo how to load from a file?

# Using the boundaries on new data
transformer = UserInputBucketer(
    binning_dict=features_bucket_mapping, return_object=False, return_boundaries=False)
transformer.fit_transform(df) # returns df
```

### Using `skorecard` with other packages

There are other packages that offer nice transformers for bucketing or modelling. Some recommendations:

- [feature-engine](https://feature-engine.readthedocs.io/en/latest/discretisers/index.html)