

def _judge_type(a, b):
    if a is None and b is not None:
        raise AttributeError
    if a is not None and b is None:
        raise AttributeError