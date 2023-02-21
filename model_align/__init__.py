# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from .ReprodLogger  import ReprodLogger
from .ReprodDiffHelper import ReprodDiffHelper
from .utils import utils
from . import compare
import paddle
import paddle.nn as pn
import paddle.nn.functional as pF
import torch.nn.functional as tF
from .utils.util import save_align_log, show_net_info, extract_grad
import torch
import torch.nn as tn
from .utils.util import get_layers
from typing import Optional, Tuple
import numpy as np
import os
import types
import copy
from .torch2paddle import convert_weight


class ModelAlign:
    """
    torch -> paddle的模型转换，需要保持paddle的模型和torch的模型结构以及
    内部的layer命名一致，如果不一致，则需要在使用前手动进行torch参数转换到paddle
    模型中，并且将 feat_align设置成False。
    Args:
        * paddle_model (paddle.nn.Layer): paddle的模型
        * torch_model (torch.nn.Module): torch的模型
        * input_data (torch.Tensor | paddle.Tensor | np.ndarray): 模型前向输入的数据
        * paddle_loss_func (Option[function | paddle.nn.Layer]): paddle的损失函数。反向对齐时需要提供。默认为None。
        * torch_loss_func (Option[function | torch.nn.Module]): torch的损失函数。反向对齐时需要提供。默认为None。
        * save_path (str): 对齐日志保存的路径，默认为 `./output`。
        * learning_rate (float): 学习率，默认为 `1e-3`。
        * diff_threshold (float): 对齐时判断的阈值。默认为 `1e-6`。
        * iters (int): 反向时迭代的iter数。默认为 5。
        * feat_align (bool): 是否需要进行网络的中间层对齐检查。如果在使用此框架之前已经将模型参数从torch转到paddle，那么此参数可以设置为False。当此参数为True时，则需要保证paddle模型和torch模型的网络层子模块的命名一致。默认为 True。
            
    Examples: 
        code-block: python
        
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
    """
    def __init__(self,
                 paddle_model: pn.Layer,
                 torch_model: tn.Module,
                 input_data: Optional[torch.Tensor | paddle.Tensor | np.ndarray],
                 paddle_loss_func: Optional[types.FunctionType | pn.Layer] = None,
                 torch_loss_func: Optional[types.FunctionType | tn.Module] = None,
                 save_path: str = './output',
                 learning_rate: float = 1e-3,
                 diff_threshold: float = 1e-6,
                 iters: int = 5,
                 feat_align: bool = True,
                 show_net: bool = True,
                 ):
        os.makedirs(os.path.join(save_path, 'data'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'align_info'), exist_ok=True)
        self.iters = iters
        self.feat_align = feat_align
        self.paddle_loss_func = paddle_loss_func
        self.torch_loss_func = torch_loss_func
        self.save_path = save_path
        paddle_model.eval()
        torch_model.eval()
        self.paddle_model = paddle_model
        self.torch_model = torch_model
        self.learning_rate = learning_rate
        self.input_data = input_data
        self.diff_threshold = diff_threshold
        self.paddle_feats = []
        self.torch_feats = []

        self._set_optimizer()
        self._set_data()
        if show_net:
            show_net_info(self.paddle_model, self.torch_model, self.save_path)
        if self.feat_align:
            self.include_buffer_layer, self.layers = get_layers(self.paddle_model, self.torch_model)
            self._hooks()

    def calculate_loss(self, 
                       paddle_output: dict | paddle.Tensor,
                       torch_output: dict | torch.Tensor,
                       paddle_input: dict = {},
                       torch_input: dict = {},
                       **kwargs,
                       ) -> Tuple[paddle.Tensor, torch.Tensor]:
        if self.paddle_loss_func:
            paddle_loss = self.paddle_loss_func(paddle_output, **paddle_input)
            torch_loss = self.torch_loss_func(torch_output, **torch_input)
            assert isinstance(paddle_loss, paddle.Tensor), f"paddle func shoudle be return type paddle.Tensor, not be type {type(paddle_loss)}"
            assert isinstance(torch_loss, torch.Tensor), f"torch func shoudle be return type torch.Tensor, not be type {type(torch_loss)}"
            
            return paddle_loss, torch_loss 

    def forward(self,
                log: bool = True,
                log_stage: str = "forward",
                ) -> Tuple[paddle.Tensor | dict | tuple, torch.Tensor | dict | tuple]:
        paddle_output = self.paddle_model(self.paddle_data)
        torch_output = self.torch_model(self.torch_data)
        if log and self.feat_align:
            if isinstance(paddle_output, paddle.Tensor):
                save_align_log(self.layers + ['output'], self.paddle_feats + [paddle_output.detach().numpy()], self.torch_feats + [torch_output.detach().cpu().numpy()], self.save_path, log_stage, diff_threshold=self.diff_threshold)
            else:
                save_align_log(self.layers, self.paddle_feats, self.torch_feats, self.save_path, log_stage, diff_threshold=self.diff_threshold)
        elif log:
            assert isinstance(paddle_output, paddle.Tensor), f"the type paddle_out shoudle be paddle.Tensor, not be type {type(paddle_output)}"
            save_align_log(['output'], [paddle_output.detach().numpy()], [torch_output.detach().cpu().numpy()], self.save_path, log_stage, diff_threshold=self.diff_threshold)
        self._clear_hooks()
        return paddle_output, torch_output

    def _hook_paddle(self, net, inputs, output):
        self.paddle_feats.append(copy.deepcopy(output.detach().numpy()))

    def _hook_torch(self, net, inputs, output):
        self.torch_feats.append(copy.deepcopy(output.detach().cpu().numpy()))

    def _hooks(self):
        for layer in self.layers:
            eval(f'self.paddle_model.{layer}').register_forward_post_hook(self._hook_paddle)
            eval(f'self.torch_model.{layer}').register_forward_hook(self._hook_torch)
    
    def _clear_hooks(self):
        self.paddle_feats.clear()
        self.torch_feats.clear()

    def _set_data(self):
        assert isinstance(self.input_data, (paddle.Tensor, torch.Tensor, np.ndarray)), f"input_data is should be paddle.Tensor, torch.Tensor or np.ndarray, not be {type(self.input_data)}"
        if isinstance(self.input_data, paddle.Tensor):
            self.paddle_data = self.input_data
            self.torch_data = torch.from_numpy(self.input_data.numpy())
        elif isinstance(self.input_data, torch.Tensor):
            self.paddle_data = paddle.to_tensor(self.input_data.numpy())
            self.torch_data = self.input_data
        elif isinstance(self.input_data, np.ndarray):
            self.paddle_data = paddle.to_tensor(self.input_data)
            self.torch_data = torch.from_numpy(self.input_data)

    def convert_weight(self):
        assert self.feat_align, f"if you want to convert_weight, the feat_align should be set True."
        convert_weight(self.paddle_model, self.torch_model, self.include_buffer_layer)

    def backward(self, 
                 paddle_input: dict = {},
                 torch_input: dict = {}):
        """
        you need provide the loss_func param
        Args:
            paddle_input (dict): paddle_loss func's param
            torch_input (dict): torch_loss func's param

        Examples:

            paddle_model = paddle_resnet18()
            torch_model = torch_resnet18()
            input_data = torch.randn((2,3,224,224))
            align = ModelAlign(paddle_model, 
                                torch_model,
                                paddle_loss_func=paddle_loss,
                                torch_loss_func=torch_loss, 
                                input_data=input_data,
                                diff_threshold=1e-6,
                                iters=2)      # 反向时迭代次数

            def paddle_loss(paddle_out, label=None):
                return pn.CrossEntropyLoss()(paddle_out, label)
            
            def torch_loss(torch_out, label=None):
                return tn.CrossEntropyLoss()(torch_out, label)

            
        """
        assert self.paddle_loss_func, f"if you want to backward align, the paddle_loss_func should not be None"
        assert self.torch_loss_func, f"if you want to backward align, the torch_loss_func should not be None"
        loss_log = []
        loss_paddle = []
        loss_torch = []
        for iter_id in range(1, self.iters + 1):
            paddle_out, torch_out = self.forward(log_stage=f"forward_iter_{iter_id}", log=True)
            paddle_loss, torch_loss = self.calculate_loss(paddle_out, torch_out, paddle_input=paddle_input, torch_input=torch_input)
            loss_log.append(f"backward_loss_iter_{iter_id}")
            loss_paddle.append(paddle_loss.detach().cpu().numpy())
            loss_torch.append(torch_loss.detach().cpu().numpy())

            self.paddle_opt.clear_grad()
            self.torch_opt.zero_grad()
            paddle_loss.backward()
            torch_loss.backward()
            names, p_feat, t_feat = extract_grad(self.paddle_model, self.torch_model, self.include_buffer_layer)
            save_align_log(names, p_feat, t_feat, self.save_path, stage=f"backward_iter_{iter_id}")

            self.paddle_opt.step()
            self.torch_opt.step()

        save_align_log(loss_log, loss_paddle, loss_torch, self.save_path, "backward_loss")

    def _set_optimizer(self):
        self.paddle_opt = paddle.optimizer.SGD(learning_rate=self.learning_rate, parameters=self.paddle_model.parameters())
        self.torch_opt = torch.optim.SGD(self.torch_model.parameters(), lr=self.learning_rate)
