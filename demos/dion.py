import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

from demos.dino.paddle_dino import vit_small as paddle_vit
from demos.dino.torch_dino import vit_small as torch_vit

import paddle
import torch

def paddle_loss(output):
    return output.mean()

def torch_loss(output):
    return output.mean()

paddle_model = paddle_vit()
torch_modle = torch_vit()

inputs_data = torch.randn((3, 3, 224, 224))

from model_align import ModelAlign
align = ModelAlign(paddle_model, torch_modle, input_data=inputs_data, paddle_loss_func=paddle_loss, torch_loss_func=torch_loss)
align.plot_weight()
# align.convert_weight()
# align.forward()
# align.backward(paddle_input={}, torch_input={})