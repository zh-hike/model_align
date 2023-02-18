from .util import _judge_type
import paddle
import torch


def convert_linear(paddle_model, torch_model):
    paddle_model.weight.set_value(paddle.to_tensor(torch_model.weight.T.detach().cpu().numpy()))
    _judge_type(paddle_model.bias, torch_model.bias)
    if paddle_model.bias is not None and torch_model.bias is not None:
        paddle_model.bias.set_value(paddle.to_tensor(torch_model.bias.detach().cpu().numpy()))
