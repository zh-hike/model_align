

def _judge_type(a, b):
    if a is None and b is not None:
        raise AttributeError("paddle_model is not have bias but torch_model have")
    if a is not None and b is None:
        raise AttributeError("paddle_model have bias but torch_model not have")