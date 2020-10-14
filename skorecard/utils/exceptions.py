class DimensionalityError(Exception):
    """Raise this when the Dimension of the numpy array or pandas DataFrame is wrong."""

    def __init__(self, message):
        """Raise this when the Dimension of the numpy array or pandas DataFrame is wrong.

        Args:
            message: (str) message when exception is raised
        """
        self.message = message


class UnknownCategoryError(Exception):
    """Raise this when an array contains a new unseen category."""

    def __init__(self, message):
        """Raise this when array contains a new unseen category.

        Args:
            message: (str) message when exception is raised
        """
        self.message = message


class NotInstalledError:
    """
    This object is used for optional dependencies.

    This allows us to give a friendly message to the user that they need to install extra dependencies as well as a link
    to our documentation page.

    Adapted from: https://github.com/RasaHQ/whatlies/blob/master/whatlies/error.py

    Example usage:

    ```python
    from skorecard.utils import NotInstalledError

    try:
        import dash_core_components as dcc
    except ModuleNotFoundError as e:
        dcc = NotInstalled("dash_core_components", "dashboard")

    dcc.Markdown() # Will raise friendly error with instructions how to solve
    ```

    Note that installing optional dependencies in a package are defined in setup.py.
    """

    def __init__(self, tool, dep=None):
        """Initialize error with missing package and reference to conditional install package.
        
        Args:
            tool (str): The name of the pypi package that is missing
            dep (str): The name of the extra_imports set (defined in setup.py) where the package is present. (optional)
        """
        self.tool = tool
        self.dep = dep

        msg = f"In order to use {self.tool} you'll need to install via;\n\n"
        if self.dep is None:
            msg += f"pip install {self.tool}\n\n"
        else:
            msg += f"pip install skorecard[{self.dep}]\n\n"

        msg += "See skorecard installation guide here: <TODO, link to our hosted docs>"
        self.msg = msg

    def __getattr__(self, *args, **kwargs):
        """Raise when accessing an attribute."""
        raise ModuleNotFoundError(self.msg)

    def __call__(self, *args, **kwargs):
        """Raise when accessing a method."""
        raise ModuleNotFoundError(self.msg)