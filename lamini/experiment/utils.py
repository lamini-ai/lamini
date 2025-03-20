def remove_non_ascii(text):
    """Remove non-ASCII characters from text input.

    This function recursively processes input text to remove non-ASCII characters.
    It handles different input types including dictionaries, strings, and lists.

    Parameters
    ----------
    text : Union[dict, str, list, Any]
        The input text to process. Can be:
        - dict: Dictionary with any type of values
        - str: String containing possible non-ASCII characters
        - list: List of any type of values
        - Any: Any other type will be returned as-is

    Returns
    -------
    Union[dict, str, list, Any]
        The processed input with non-ASCII characters removed:
        - dict: Dictionary with processed values
        - str: ASCII-only string
        - list: List with processed values
        - Any: Unmodified input for other types

    Examples
    --------
    >>> remove_non_ascii("Hello™ World®")
    'Hello World'
    >>> remove_non_ascii({"key": "value™"})
    {'key': 'value'}
    >>> remove_non_ascii(["text™", "more©"])
    ['text', 'more']
    """
    if isinstance(text, dict):
        return {
            key_: remove_non_ascii(text[key_])
            for key_ in text
        }
    elif isinstance(text, str):
        return text.encode("ascii", "ignore").decode()
    elif isinstance(text, list):
        return [remove_non_ascii(item) for item in text]
    else:
        return text