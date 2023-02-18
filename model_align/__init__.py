import paddle
import paddle.nn as pn
import paddle.nn.functional as pF
import torch.nn.functional as tF
from .util import save_align_log
import torch
import torch.nn as tn
from .util import get_layers
from typing import Optional, Tuple
import numpy as np
import os
import types
from .torch2paddle import convert_weight


class ModelAlign:
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
                 ):
        os.makedirs(os.path.join(save_path, 'data'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'align_info'), exist_ok=True)
        self.iters = iters
        self.paddle_loss_func = paddle_loss_func
        self.torch_loss_func = torch_loss_func
        assert self.paddle_loss_func
        assert self.torch_loss_func
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
        self.layers = get_layers(self.paddle_model, self.torch_model)
        self._hooks()

    def calculate_loss(self, 
                       paddle_output: Optional[dict | paddle.Tensor],
                       torch_output: Optional[dict | torch.Tensor],
                       paddle_input: dict = {},
                       torch_input: dict = {},
                       **kwargs,
                       ) -> Tuple[paddle.Tensor, torch.Tensor]:
        
        if self.paddle_loss_func:
            return self.paddle_loss_func(paddle_output, **paddle_input), self.torch_loss_func(torch_output, **torch_input)

    def forward(self,
                log: bool = True,
                log_stage: str = "forward",
                ) -> Tuple[Optional[paddle.Tensor | dict | tuple], Optional[torch.Tensor | dict | tuple]]:
        paddle_output = self.paddle_model(self.paddle_data)
        torch_output = self.torch_model(self.torch_data)
        if log:
            save_align_log(self.layers, self.paddle_feats, self.torch_feats, self.save_path, log_stage, diff_threshold=self.diff_threshold)
        print(len(self.paddle_feats), len(self.torch_feats))
        self._clear_hooks()
        return paddle_output, torch_output

    def _hook_paddle(self, net, inputs, output):
        self.paddle_feats.append(output.detach().numpy())

    def _hook_torch(self, net, inputs, output):
        self.torch_feats.append(output.detach().cpu().numpy())

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
        convert_weight(self)

    def backward(self, **kwargs):
        loss_log = []
        loss_paddle = []
        loss_torch = []
        for iter_id in range(1, self.iters + 1):
            paddle_out, torch_out = self.forward(log_stage=f"backward_iter_{iter_id}")
            paddle_loss, torch_loss = self.calculate_loss(paddle_out, torch_out, **kwargs)
            loss_log.append(f"backward_loss_iter_{iter_id}")
            loss_paddle.append(paddle_loss.detach().cpu().numpy())
            loss_torch.append(torch_loss.detach().cpu().numpy())
        save_align_log(loss_log, loss_paddle, loss_torch, self.save_path, "backward_loss")

    def _set_optimizer(self):
        self.paddle_opt = paddle.optimizer.SGD(learning_rate=self.learning_rate, parameters=self.paddle_model.parameters())
        self.torch_opt = torch.optim.SGD(self.torch_model.parameters(), lr=self.learning_rate)
