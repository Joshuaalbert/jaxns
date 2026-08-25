"""Lazy imports for opt-in user-interface features."""


def import_matplotlib():
    """Import Matplotlib's pyplot module for plotting entry points.

    Returns:
        The imported ``matplotlib.pyplot`` module.

    Raises:
        ImportError: If the plotting extra is not installed correctly.
    """
    try:
        from matplotlib import pyplot
    except ImportError as error:
        raise ImportError(
            "JAXNS plotting requires the optional plotting dependencies. "
            "Install them with `pip install 'jaxns[plotting]'`."
        ) from error
    return pyplot
