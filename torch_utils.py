import os

import torch
import numpy as np
def split_tensor_to_complex(tensor):
    """
    将一个 tensor 分成两部分，一部分为实部，一部分为虚部，并组合成复数 tensor。
    
    参数:
        tensor: 输入 tensor，长度必须是偶数。
        
    返回:
        复数 tensor，每个元素由前半部分的实部和后半部分的虚部组成。
    """
    # 确保 tensor 长度是偶数
    if tensor.shape[-1] % 2 != 0:
        raise ValueError("Tensor length must be even to split into real and imaginary parts.")
    
    # 分离实部和虚部
    mid_point = tensor.shape[-1] // 2
    real_part = tensor[..., :mid_point]
    imag_part = tensor[..., mid_point:]
    
    # 组合为复数 tensor
    complex_tensor = real_part + 1j * imag_part
    return complex_tensor

def split_and_recombine_tensor(complex_tensor):
    """
    将一个 tensor 分成两部分，一部分为实部，一部分为虚部，
    然后将其组合回到复数形式的 tensor，并且分离出实部和虚部的 tensor。

    参数:
        complex_tensor: 输入 complex_tensor

    返回:
        real_tensor: 复数 complex_tensor 的实部。
        imag_tensor: 复数 complex_tensor 的虚部。
    """

    # 提取实部和虚部，并分别转换为单独的 tensor
    real_tensor = torch.real(complex_tensor)
    imag_tensor = torch.imag(complex_tensor)
    
    restored_tensor = torch.cat((real_tensor, imag_tensor), dim=-1)

    return restored_tensor

def awgn_channel(signal, snr_linear):
    """
    在AWGN信道中添加噪声。
    
    参数:
        signal: 输入信号 (torch.Tensor)。
        snr_linear: 信噪比（线性比例）。
        
    返回:
        加上AWGN噪声后的信号 (torch.Tensor)。
    """
    print(snr_linear)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    noise_power = signal_power / snr_linear
    noise_power = torch.clamp(noise_power, min=1e-10)
    # 生成复数噪声
    print(signal_power)
    print(noise_power)
    mean = torch.zeros(signal.shape, device=signal.device)  # 与 signal 形状相同的零张量
    std = torch.sqrt(noise_power / 2).expand_as(mean).to(signal.device)  # 将标准差扩展为与 mean 相同的形状
    noise = torch.normal(mean, std).to(signal.device) + 1j * torch.normal(mean, std).to(signal.device)

    # noise = torch.normal(0, torch.sqrt(noise_power / 2), signal.shape).to(signal_power.device) + 1j * torch.normal(0, torch.sqrt(noise_power / 2), signal.shape).to(signal_power.device)
    return signal + noise

def rayleigh_to_awgn(complex_tensor, snr_linear):
    """
    将瑞利或MIMO信道下的信号转换为AWGN信道的等效形式。
    
    参数:
        complex_tensor: 输入信号 (torch.Tensor)。
        snr_linear: 信噪比（线性比例）。
        
    返回:
        经过均衡后的信号，类似于AWGN信道中的输出 (torch.Tensor)。
    """
    # 生成瑞利信道增益（复数形式）
    # 生成信道增益恒为 1 的信道矩阵
    sigma_h = 1 ## 信道增益
    real_part = torch.normal(0, sigma_h  / np.sqrt(2), complex_tensor.shape).to(complex_tensor.device)
    imag_part = torch.normal(0, sigma_h  / np.sqrt(2), complex_tensor.shape).to(complex_tensor.device)
    h = real_part + 1j * imag_part
    # 通过瑞利信道
    signal = h * complex_tensor

    # 添加AWGN噪声
    signal_power = torch.mean(torch.abs(signal) ** 2)
    noise_power = signal_power / snr_linear

    mean = torch.zeros(signal.shape, device=signal.device)  # 与 signal 形状相同的零张量
    std = torch.sqrt(noise_power / 2).expand_as(mean).to(signal.device)  # 将标准差扩展为与 mean 相同的形状
    noise = torch.normal(mean, std).to(signal.device) + 1j * torch.normal(mean, std).to(signal.device)

    # noise = torch.normal(0, torch.sqrt(noise_power / 2), signal.shape).to(signal.device) + 1j * torch.normal(0, torch.sqrt(noise_power / 2), signal.shape).to(signal.device)
    received_signal = signal + noise
    # 零强度均衡，恢复为AWGN信道
    # equalized_signal = received_signal / h
    # print('zf equalization')

    # 计算 MMSE 均衡器系数
    Px = torch.mean(torch.abs(complex_tensor) ** 2)  # 计算发送信号功率
    W = h.conj() / (torch.abs(h) ** 2 + noise_power / Px)

    # 应用 MMSE 均衡器
    equalized_signal = W * received_signal

    return equalized_signal

def mimo_channel_to_awgn(complex_tensor, snr_linear):
    """
    将 2x2 MIMO 信道通过 SVD 分解，转化为独立的 AWGN 信道。

    参数:
        complex_tensor: 输入复数信号张量，形状为 [B, ..., 2]。
        H: 2x2 MIMO 信道矩阵，形状为 (2, 2)。
        snr_linear: 信噪比。

    返回:
        signal_decoded: 解码后的信号，类似于 AWGN 信道输出，形状与输入 complex_tensor 相同。
    """
    b, c, w, h = complex_tensor.shape
    mid_point = complex_tensor.shape[0] // 2
    x1 = complex_tensor[:mid_point, ...].unsqueeze(-1)
    x2 = complex_tensor[mid_point:, ...].unsqueeze(-1)

    ######  部分cuda环境下, 因精度问题, 导致SVD算法部分失真
    signal = torch.cat((x1, x2), dim = -1).unsqueeze(-1).cpu()
    # signal = torch.randn((1, 4, 64, 32, 2, 1), dtype=torch.cfloat).to(complex_tensor.device)  # 2x1 输入信号

    # 1. 对 MIMO 信道矩阵 H 进行 SVD 分解,
    random_matrix = torch.randn(2, 2, dtype=torch.cfloat).to(signal.device)
    while torch.det(random_matrix).abs() < 1e-6:
        random_matrix = torch.randn(2, 2, dtype=torch.cfloat).to(signal.device)

    # 1. 生成奇异值恒为 1 的信道矩阵, 确保子信道的SNR一致
    H, R = torch.linalg.qr(random_matrix)
    H = H * (R.diag() / R.diag().abs())
    U, S, Vh = torch.linalg.svd(H)
    S_matrix = torch.diag(S).cfloat()

    # 2. 预编码：将输入信号转换到 SVD 的右奇异向量空间
    x_encoded = torch.matmul(Vh.conj().T, signal)

    # 3. 添加 AWGN 噪声
    signal_power = torch.mean(torch.abs(x_encoded) ** 2)
    noise_power = signal_power / snr_linear

    mean = torch.zeros(x_encoded.shape, device=x_encoded.device)  # 与 signal 形状相同的零张量
    std = torch.sqrt(noise_power / 2).expand_as(mean).to(x_encoded.device)  # 将标准差扩展为与 mean 相同的形状
    noise = torch.normal(mean, std).to(x_encoded.device) + 1j * torch.normal(mean, std).to(x_encoded.device)

    # noise = torch.normal(0, torch.sqrt(noise_power / 2), x_encoded.shape).to(x_encoded.device) + 1j * torch.normal(0, torch.sqrt(noise_power / 2), x_encoded.shape).to(x_encoded.device)
    y_noisy = torch.matmul(H, x_encoded) + noise

    # 3. 验证重构矩阵是否与原始矩阵相同 
    # print("重构是否与原始矩阵一致:", torch.allclose(torch.matmul(S_matrix, signal), torch.matmul(U.conj().T, y_noisy), atol=1e-6))
    y_decoded = torch.matmul(U.conj().T, y_noisy).to(complex_tensor.device)

    x1_decoded = y_decoded.squeeze(-1)[..., 0]
    x2_decoded = y_decoded.squeeze(-1)[..., 1]
    signal_decoded = torch.cat((x1_decoded, x2_decoded), dim = 0)
    return signal_decoded

def form_signal_to_tensor(tensor, snr_db, channel_type): #form_signal_to_tensor
    complex_signal = split_tensor_to_complex(tensor)
    snr_linear = 10 ** (snr_db / 10)

    if channel_type == 'awgn':
        equalized_signal = awgn_channel(complex_signal, snr_linear)
    elif channel_type == 'fading':
        equalized_signal = rayleigh_to_awgn(complex_signal, snr_linear)
    elif channel_type == 'mimo':
        equalized_signal = mimo_channel_to_awgn(complex_signal, snr_linear)

    received_signal = split_and_recombine_tensor(equalized_signal)

    return received_signal

def compute_psnr(image1,image2,data_range=1):
    batch_size = image1.shape[0]
    psnr = 0
    for batch_idx in range(batch_size):     
        # print(image1[batch_idx].shape)   
        img1 = image1[batch_idx].cpu().detach().numpy()
        img2 = image2[batch_idx].cpu().detach().numpy()
        error = np.mean((img1 - img2) ** 2, dtype=np.float64)
        psnr += 10 * np.log10((data_range ** 2) / error)
    return np.round(psnr, decimals=2) 

def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(device='', apex=False):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        cuda_str = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                cuda_str = ' ' * len(cuda_str)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (cuda_str, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    bias=True)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0)).cuda()
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(b_conv + b_bn)

        return fusedconv


def model_info(model, report='summary'):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report is 'full':
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))


def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    import pretrainedmodels  # https://github.com/Cadene/pretrained-models.pytorch#torchvision
    model = pretrainedmodels.__dict__[name](num_classes=1000, pretrained='imagenet')

    # Display model properties
    for x in ['model.input_size', 'model.input_space', 'model.input_range', 'model.mean', 'model.std']:
        print(x + ' =', eval(x))

    # Reshape output to n classes
    filters = model.last_linear.weight.shape[1]
    model.last_linear.bias = torch.nn.Parameter(torch.zeros(n))
    model.last_linear.weight = torch.nn.Parameter(torch.zeros(n, filters))
    model.last_linear.out_features = n
    return model
