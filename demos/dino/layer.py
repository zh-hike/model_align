from .init import reset_initialized_parameter
import paddle.nn as nn

__all__ = ['LayerNorm',
           'Linear',
           'Conv2D']

class LayerNorm(nn.LayerNorm):
    def __init__(self, *warg, **kwargs):
        super(LayerNorm, self).__init__(*warg, **kwargs)
        reset_initialized_parameter(self)

class Linear(nn.Linear):
    def __init__(self, *warg, **kwargs):
        super(Linear, self).__init__(*warg, **kwargs)
        reset_initialized_parameter(self)

class Conv2D(nn.Conv2D):
    def __init__(self, *warg, **kwargs):
        super(Conv2D, self).__init__(*warg, **kwargs)
        reset_initialized_parameter(self)