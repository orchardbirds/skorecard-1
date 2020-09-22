"""
Tests all the codeblocks in the docstrings, making sure they execute.

This approach is adapted from, and explained in:
https://calmcode.io/docs/epic.html

Try it out with:

> pytest tests/test_docstring.py --verbose

"""

import pytest
import skorecard.apps
import skorecard.linear_model
import skorecard.preprocessing
import skorecard.pipeline
import skorecard.metrics
import skorecard.bucket_mapping
import skorecard.utils


# List of all classes and functions we want tested
CLASSES_TO_TEST = [
    skorecard.apps.ManualBucketerApp,
    skorecard.linear_model.LogisticRegression,
    skorecard.pipeline.ColumnSelector,
    skorecard.preprocessing.SimpleBucketTransformer,
    skorecard.preprocessing.AgglomerativeBucketTransformer,
    skorecard.preprocessing.QuantileBucketTransformer,
    skorecard.preprocessing.TreeBucketTransformer,
    skorecard.preprocessing.CatBucketTransformer,
    skorecard.bucket_mapping.FeaturesBucketMapping,
    skorecard.bucket_mapping.BucketMapping,
]
FUNCTIONS_TO_TEST = [
    skorecard.utils.assure_numpy_array,
    skorecard.utils.reshape_1d_to_2d,
    skorecard.utils.DimensionalityError,
]


def get_public_methods(cls_ref):
    """Helper test function, gets all public methods in a class.
    """
    return [m for m in dir(cls_ref) if m == "__init__" or not m.startswith("_")]


def get_test_pairs(classes_to_test):
    """Helper test function, get tuples with class and public method.
    """
    test_pairs = []
    for cls_ref in classes_to_test:
        for meth_ref in get_public_methods(cls_ref):
            test_pairs.append((cls_ref, meth_ref))
    return test_pairs


def handle_docstring(doc, indent):
    """
    Check python code in docstring.

    This function will read through the docstring and grab
    the first python code block. It will try to execute it.
    If it fails, the calling test should raise a flag.
    """
    if not doc:
        return
    start = doc.find("```python\n")
    end = doc.find("```\n")
    if start != -1:
        if end != -1:
            code_part = doc[(start + 10) : end].replace(" " * indent, "")
            print(code_part)
            exec(code_part)


# Test every method in a list of selected classes
@pytest.mark.parametrize("clf_ref,meth_ref", get_test_pairs(CLASSES_TO_TEST))
def test_skorecard_method_docstrings(clf_ref, meth_ref):
    """
    Take the docstring of every method (m) on the class (c).

    The test passes if the usage examples causes no errors.
    """
    handle_docstring(getattr(clf_ref, meth_ref).__doc__, indent=8)


# Test class docstrings
@pytest.mark.parametrize("m", CLASSES_TO_TEST)
def test_class_docstrings(m):
    """
    Take the docstring of every method on a given class.

    The test passes if the usage examples causes no errors.
    """
    handle_docstring(m.__doc__, indent=4)


# Test every function in a module
@pytest.mark.parametrize("m", FUNCTIONS_TO_TEST)
def test_function_docstrings(m):
    """
    Take the docstring of every function.

    The test passes if the usage examples causes no errors.
    """
    handle_docstring(m.__doc__, indent=4)
