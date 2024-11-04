import inspect
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('jaxns')


def get_grandparent_info(relative_depth: int = 7):
    """
    Get the file, line number and function name of the caller of the caller of this function.

    Args:
        relative_depth: the number of frames to go back from the caller of this function. Default is 6. Should be
        enough to get out of a jax.tree.map call.

    Returns:
        str: a string with the file, line number and function name of the caller of the caller of this function.
    """
    # Get the grandparent frame (caller of the caller)
    s = []
    for depth in range(1, min(1 + relative_depth, len(inspect.stack()) - 1) + 1):
        caller_frame = inspect.stack()[depth]
        caller_file = caller_frame.filename
        caller_line = caller_frame.lineno
        caller_func = caller_frame.function
        s.append(f"{os.path.basename(caller_file)}:{caller_line} in {caller_func}")
    s = s[::-1]
    s = f"at {' -> '.join(s)}"
    return s
