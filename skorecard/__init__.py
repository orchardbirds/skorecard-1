# from sklearn.pipeline import Pipeline
# from skorecard.pipeline import get_features_bucket_mapping

# Monkey patch Pipeline class
# This adds a property method you can use to
# find the bucketing information from bucket transformers
# anywhere in the sklearn pipeline.
# Pipeline.features_bucket_mapping_ = property(lambda self: get_features_bucket_mapping(self))
