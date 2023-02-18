from .convert_layer import convert_bn, convert_conv, convert_linear, convert_ln
import paddle.nn as pn
import torch.nn as tn
import paddle


def get_convert_func(paddle_model, torch_model, layer):
    if isinstance(eval(f"paddle_model.{layer}"), pn.Linear):
        return convert_linear
    if isinstance(eval(f"paddle_model.{layer}"), (pn.Conv1D, pn.Conv2D, pn.Conv3D, pn.Conv1DTranspose, pn.Conv2DTranspose, pn.Conv3DTranspose)):
        return convert_conv
    if isinstance(eval(f"paddle_model.{layer}"), pn.LayerNorm):
        return convert_ln
    if isinstance(eval(f"paddle_model.{layer}"), (pn.BatchNorm1D, pn.BatchNorm2D, pn.BatchNorm3D)):
        return convert_bn
    return None

def convert_weight(paddle_model, torch_model, layers):
    paddle_model = paddle_model
    torch_model = torch_model
    for layer in layers:
        if not isinstance(eval(f"paddle_model.{layer}"), pn.Layer):
            eval(f"paddle_model.{layer}").set_value(paddle.to_tensor(eval(f"torch_model.{layer}").detach().cpu().numpy()))
            continue
        convert_func = get_convert_func(paddle_model, torch_model, layer)
        assert convert_func is not None, f"can't convert {layer}"
        try:
            convert_func(eval(f'paddle_model.{layer}'), eval(f'torch_model.{layer}'))
        except:
            raise AssertionError(f"the layer {layer} is different between paddle_model and torch_model")
        
        

