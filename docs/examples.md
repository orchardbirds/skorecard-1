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


# Manual tweaking (optional)
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
    <preproc>,
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