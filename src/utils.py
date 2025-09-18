def sanitize_str(s):
    """
    Sanitize a string to be a valid filename.

    Sanitization involves replacing spaces, hyphens, parentheses,
    and slashes with underscores.    

    Parameters
    ----------
    s : str
        The input string to sanitize.

    Returns
    -------
    str
        The sanitized string.

    """

    return s.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('/', '_')
