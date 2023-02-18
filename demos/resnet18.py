import paddle
import torch
import paddle.nn as pn
import torch.nn as tn
from model_align import ModelAlign
from paddle.vision.models import resnet18 as paddle_resnet18
from torchvision.models import resnet18 as torch_resnet18

def paddle_loss(paddle_out, label=None):
        return pn.CrossEntropyLoss()(paddle_out, label)
    
def torch_loss(torch_out, label=None):
    return tn.CrossEntropyLoss()(torch_out, label)

paddle.set_device('cpu')

paddle_model = paddle_resnet18()
torch_model = torch_resnet18()
input_data = torch.randn((2,3,224,224))
align = ModelAlign(paddle_model, 
                   torch_model,
                   paddle_loss_func=paddle_loss,
                   torch_loss_func=torch_loss, 
                   input_data=input_data,
                   diff_threshold=100,
                   save_path="./output/resnet18",
                   iters=3,
                   feat_align=True)
align.convert_weight()
align.forward()
torch_input = torch.randint(0, 100, (2,))
paddle_input = paddle.to_tensor(torch_input.numpy())
align.backward(paddle_input={'label':paddle_input}, 
                torch_input={'label':torch_input})