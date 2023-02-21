from matplotlib import pyplot as plt
import os

def plot(data, save_file):
    import seaborn as sns
    fig = plt.figure()
    sns.histplot(data.reshape(-1), kde=True, bins=30)
    plt.savefig(save_file, dpi=300)

def plot_weight_distributed(paddle_model, torch_model, layers, save_path):
    os.makedirs(os.path.join(save_path, 'weight_distribute'), exist_ok=True)
    for layer in layers:
        print(f"plot {layer}...")
        paddle_net = eval(f"paddle_model.{layer}")
        torch_net = eval(f"torch_model.{layer}")
        paddle_weight = paddle_net.weight.numpy().reshape(-1)
        torch_weight = torch_net.weight.detach().cpu().numpy().reshape(-1)
        paddle_save_file = os.path.join(save_path, 'weight_distribute', f'{layer}_weight_paddle.png')
        torch_save_file = os.path.join(save_path, 'weight_distribute', f'{layer}_weight_torch.png')
        plot(paddle_weight, paddle_save_file)
        plot(torch_weight, torch_save_file)

        if paddle_net.bias is not None:
            paddle_bias = paddle_net.bias.numpy().reshape(-1)
            torch_bias = torch_net.bias.detach().cpu().numpy().reshape(-1)
            paddle_save_file = os.path.join(save_path, 'weight_distribute', f'{layer}_bias_paddle.png')
            torch_save_file = os.path.join(save_path, 'weight_distribute', f'{layer}_bias_torch.png')
            plot(paddle_bias, paddle_save_file)
            plot(torch_bias, torch_save_file)



