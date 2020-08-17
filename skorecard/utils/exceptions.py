class DimensionalityError(Exception):
    """Raise this when the Dimension of the numpy array or pandas DataFrame is wrong."""

    def __init__(self, message):
        """Raise this when the Dimension of the numpy array or pandas DataFrame is wrong.

        Args:
            message: (str) message when exception is raised
        """
        self.message = message
