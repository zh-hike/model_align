from __future__ import annotations
import re
from typing import Optional
import numpy as np
import os
from reprod_log import ReprodLogger, ReprodDiffHelper
import paddle.nn as pn
import torch.nn as tn


def convert_key_to_layer_name(key: str) -> str:

    def convert(match) ->str:
        age = match.group()
        return '['+age.strip('.')+'].'
    
    layer_name = re.sub('(\\.[0-9]\\.+)', convert, key+'.').strip('.')
    return layer_name

def check_diff(paddle_npy: str,
               torch_npy: str,
               save_path: str,
               stage: str,
               diff_threshold: float = 1e-6):
    checker = ReprodDiffHelper()
    paddle_data = checker.load_info(paddle_npy)
    torch_npy = checker.load_info(torch_npy)
    checker.compare_info(paddle_data, torch_npy)
    checker.report(path=os.path.join(save_path, 'align_info', stage+'.txt'),
                   diff_threshold=diff_threshold)

def save_align_log(names: str | list, 
                   paddle_feat: np.ndarray | list, 
                   torch_feat: np.ndarray | list,
                   save_path: str,
                   stage: str,
                   diff_threshold: float = 1e-6) -> None:
    """
    
    stage: parameters, forward, backward
    save_path + paddle+stage     为保存的文件名
    """
    paddle_log = ReprodLogger()
    torch_log = ReprodLogger()
    paddle_log.clear()
    torch_log.clear()
    if not isinstance(names, list):
        names = [names]
    if not isinstance(paddle_feat, list):
        paddle_feat = [paddle_feat]
    if not isinstance(torch_feat, list):
        torch_feat = [torch_feat]
    assert len(names) == len(paddle_feat) == len(torch_feat), "number of names, paddle_feat, torch_feat is not equal"
    for name, pf, tf in zip(names, paddle_feat, torch_feat):
        paddle_log.add(name, pf)
        torch_log.add(name, tf)
    
    paddle_log.save(os.path.join(save_path, 'data/', 'paddle_'+stage))
    torch_log.save(os.path.join(save_path, 'data/', 'torch_'+stage))
    check_diff(os.path.join(save_path, 'data/', 'paddle_'+stage+'.npy'),
               os.path.join(save_path, 'data/', 'torch_'+stage+'.npy'),
               save_path,
               stage,
               diff_threshold=diff_threshold)
    

def get_layers(paddle_model: pn.Layer, torch_model: tn.Module):
        paddle_layer_name = {}.fromkeys([k[:k.rindex('.')] for k in paddle_model.state_dict().keys()]).keys()
        torch_layer_name = {}.fromkeys([k[:k.rindex('.')] for k in torch_model.state_dict().keys()]).keys()

        paddle_layer_name = [convert_key_to_layer_name(k) for k in paddle_layer_name]
        torch_layer_name = [convert_key_to_layer_name(k) for k in torch_layer_name]
        assert len(paddle_layer_name) == len(torch_layer_name), "the number of paddle_model'layer is not equal torch_model'layer"
        for p_name, t_name in zip(paddle_layer_name, torch_layer_name):
            assert p_name == t_name, f"paddle_model's layer name {p_name} is must equal torch_model's layer name {t_name}"
        return paddle_layer_name