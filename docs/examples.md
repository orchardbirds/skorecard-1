# Example use cases

## Manually finding bucketers

```python
X, y = load_data()

# Experiment with automatic bucketeers
# Or, use pipe = Skorecard().pipeline
pipe = Pipeline(
    <features & transformers>,
    BucketSmoother()
)
pipe.fit(X, y)

# Grid searching
# TODO: Write a Bucketer class that also takes a bucketer classname as input
# Then you can search over n_bins and bucketers.

# Optimizing bucketing pipeline
# Per feature, you can have different n_bins, and transformers
# Write a for loop that does exhaustive search (univariately per feature using x and y).
# use tqdm, implement a budget?, it should return a best_pipeline.
# also parameter on which to optimize for. AUC? WoE? IV?

# Manual tweaking (optional)
# TODO, write this manual feature mapping
features_bucket_mapping = get_feature_mapping(pipe)

print(features_bucket_mapping)
features_bucket_mapping.save_to_yml('buckets.yml')

app = ManualBucketingApp(X, y, features_bucket_mapping) # existing feature mapping is optional.
app.run_server()
features_bucket_mapping = app.features_bucket_mapping

# Reporting
from skorecard.reporting import .....
report(X, y, features_bucket_mapping) # HTML report? plot? excel output? pandas profiler? how to put into TMD?

# Actual modelling pipeline
features_bucket_mapping = FeatureBucketMapping('buckets.yml')

pipe = Pipeline(
    PandasDFTransformer(
        <preproc>,
    ),
    ManualBucketTransformer(features_bucket_mapping), # EITHER THIS. when usiung manual
    bucketing_pipeline(), # OR THIS. when defined yourself
    <model>
)
pipe.fit(X,y).predict_proba()
```

## Fully automatic

```python
X, y = load_data()

# Experiment with automatic bucketeers
# Todo think about the default pipeline.
model = Skorecard(...,num_cols = [], cat_cols = []) # not num_cols and cat_cols not specified, auto-detect + raise warning.
model.fit(X, y)

model.pipeline # printed version
Pipeline(..........)

# More reporting on the model

# Reject Inference

```