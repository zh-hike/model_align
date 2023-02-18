from .util import _judge_type
import paddle
import torch

def convert_bn(paddle_model, torch_model):
    paddle_model.weight.set_value(paddle.to_tensor(torch_model.weight.detach().cpu().numpy()))
    _judge_type(paddle_model.bias, torch_model.bias)
    if paddle_model.bias is not None and torch_model.bias is not None:
        paddle_model.bias.set_value(paddle.to_tensor(torch_model.bias.detach().cpu().numpy()))
    
    paddle_model._mean.set_value(paddle.to_tensor(torch_model.running_mean.detach().cpu().numpy()))
    paddle_model._variance.set_value(paddle.to_tensor(torch_model.running_var.detach().cpu().numpy()))