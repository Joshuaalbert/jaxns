"""Lazy imports for opt-in user-interface features."""


def import_matplotlib():
    """Import Matplotlib's pyplot module for plotting entry points.

    Returns:
        The imported ``matplotlib.pyplot`` module.

    Raises:
        ImportError: If the default installation is incomplete.
    """
    try:
        from matplotlib import pyplot
    except ImportError as error:
        raise ImportError(
            "JAXNS plotting requires Matplotlib, which is part of the default "
            "installation. Reinstall it with `pip install jaxns`."
        ) from error
    return pyplot
