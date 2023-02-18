#参数使用说明

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

> 使用方法请参考范例运行的脚本 [这里](../demos/run.sh)。
>
