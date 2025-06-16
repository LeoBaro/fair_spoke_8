class DictWrapper:
    """Helper class to convert dictionary to object-like access."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, DictWrapper(value) if isinstance(value, dict) else value)