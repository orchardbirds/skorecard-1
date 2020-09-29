# SCRATCHPAD, temporary.

## Agenda meeting 30 sept

- Discuss changes made
- Go through user guides
- New naming scheme? bucketers vs preprocessing.<name>Encoder ?
    - Also possible. variable discretizer, or Binner?
    - Split numerical bucketers vs categorical bucketers
- TODO issue list
    - reporting modules?
    - grid search / optimzer stuff? 
    - manual bucketing app


```python
from skorecard import datasets
from sklearn.pipeline import make_pipeline
from feature_engine.discretisers import UserInputDiscretiser
import feature_engine.discretisers as dsc

pipe = make_pipeline(
    dsc.EqualWidthDiscretiser(bins=4, variables=['EDUCATION']),
    dsc.EqualWidthDiscretiser(bins=7, variables=['LIMIT_BAL']),
)
pipe.fit_transform(df) # returns df

# Getting the boundaries
def get_bin_dicts(pipe):

    feature_bucket_mapping = {}

    for trx in pipe.steps:
        for key, value in trx[1].binner_dict_.items():
           feature_bucket_mapping[key] = value

    return feature_bucket_mapping

user_dict = get_bin_dicts(pipe)

# Using the boundaries on new data
transformer = UserInputDiscretiser(
    binning_dict=user_dict, return_object=False, return_boundaries=False)
transformer.fit_transform(df) # returns df

# using a probatus binner
from feature_engine.wrappers import SklearnTransformerWrapper
from probatus.binning import QuantileBucketer

bucketer = SklearnTransformerWrapper(transformer = QuantileBucketer(bin_count = 5),
                                    variables = ['LIMIT_BAL', 'BILL_AMT1'])

# fit the wrapper + SimpleImputer
bucketer.fit(df)
```