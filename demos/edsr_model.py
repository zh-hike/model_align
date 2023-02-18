import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))
from demos.edsr.paddle_edsr.paddle_edsr import EDSR as pEDSR
from demos.edsr.torch_edsr.edsr import EDSR as tEDSR
from model_align import ModelAlign
import paddle
import torch
import paddle.nn as pn
import torch.nn as tn

def paddle_loss(paddle_out, label):
    return pn.functional.mse_loss(paddle_out, label)

def torch_loss(torch_out, label):
    return tn.functional.mse_loss(torch_out, label)

pmodel = pEDSR()
tmodel = tEDSR()

torch_label = torch.randn((3, 3, 448, 448))
paddle_label = paddle.to_tensor(torch_label.detach().cpu().numpy())

align = ModelAlign(pmodel, 
                   tmodel, 
                   paddle.randn((3,3,224,224)), 
                   paddle_loss_func=paddle_loss,
                   torch_loss_func=torch_loss,
                   save_path='./output/edsr',
                   iters=3)
align.convert_weight()
align.forward()
align.backward(paddle_input={'label':paddle_label}, torch_input={'label':torch_label})
