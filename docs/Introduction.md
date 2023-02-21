#参数使用说明

## ModelAlian (class)
* \_\_init\_\_()
* * > paddle_model (paddle.nn.Layer): paddle的模型
* * > torch_model (torch.nn.Module): torch的模型
* * > input_data (torch.Tensor | paddle.Tensor | np.ndarray): 模型前向输入的数据
* * > paddle_loss_func (Option[function | paddle.nn.Layer]): paddle的损失函数。反向对齐时需要提供。默认为`None`。
* * > torch_loss_func (Option[function | torch.nn.Module]): torch的损失函数。反向对齐时需要提供。默认为 `None`。
* * > save_path (str): 对齐日志保存的路径，默认为 `./output`。
* * > learning_rate (float): 学习率，默认为 `1e-3`。
* * > diff_threshold (float): 对齐时判断的阈值。默认为 `1e-6`。
* * > iters (int): 反向时迭代的iter数。默认为 `5`。
* * > feat_align (bool): 是否需要进行网络的中间层对齐检查。如果在使用此框架之前已经将模型参数从torch转到paddle，那么此参数可以设置为False。当此参数为True时，则需要保证paddle模型和torch模型的网络层子模块的命名一致。默认为 `True`。

*  forward(log: bool=True, log_stage: str="forward")  `前向对齐`
* * > log (bool): 是否打印前向对齐的日志，默认为 `True`。
* * > log_stage (str): 前向日志保存的文件命名， 默认为 `forward`。
* convert_weight() `权重转换`
* * > 无
* backward(paddle_input: dict=None, torch_input: dict=None) `反向对齐`
* * > paddle_input (dict): paddle的loss函数的输入。默认为 `{}`。
* * > torch_input (dict): torch的loss函数的输入。默认为 `{}`。

----
utils.init.reset_initialized_parameter(paddle_model, include_self: bool=True)
重新设置paddle模型权重。
* * > paddle_model

> 使用方法请参考范例运行的脚本 [这里](../demos/run.sh)。
>
