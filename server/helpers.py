def load_file(path, default=None):
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError as e:
        if default is None:
            raise e
        else:
            return default
