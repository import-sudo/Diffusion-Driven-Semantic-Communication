"""make variations of input image"""

import argparse, os
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from torch.utils.data import Dataset
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder
from scripts.txt2img import put_watermark
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import DDPM
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch_utils import *
import random
import time
class ImageNetDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: str, root_path: str, transform=None):
        with open(images_path) as f:
            self.images_path = f.readlines()
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        file_name = self.images_path[item]
        image_path = self.root_path + file_name.split(' ')[0]
        label = file_name.split(' ')[1]
        img = Image.open(image_path).convert('RGB') 
        # RGB为彩色图片，L为灰度图片
        img = img.resize((768, 768), resample=Image.LANCZOS)
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(image_path))
        label = int(label)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, file_name.split(' ')[0]

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels, file_names = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels, file_names

class LSUNDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, root_path: str, n_samples: int, transform=None):
        self.images_path = os.listdir(root_path)
        self.transform = transform
        self.root_path = root_path
        self.n_samples = n_samples

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image_path = self.root_path + self.images_path[item]
        img = load_img(image_path)
        img = repeat(img, '1 ... -> b ...', b=self.n_samples)

        return img, self.images_path[item]

    @staticmethod
    def collate_fn(batch):
        images, file_names = tuple(zip(*batch))
        images = torch.cat(images, dim=0)
        # images = torch.stack(images, dim=0)
        return images, file_names

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

class encoder_unit(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(encoder_unit, self).__init__()
        self.block1 = nn.Sequential(
                nn.Conv2d(input_channel, input_channel* 2, kernel_size=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(input_channel* 2),

                nn.Conv2d(input_channel* 2, input_channel * 2, kernel_size=3, padding=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(input_channel* 2),

                nn.Conv2d(input_channel * 2, input_channel, kernel_size=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(input_channel),
            )

        self.conv = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(output_channel),
            )

    def forward(self, x):
        # residual = x
        x = self.block1(x) + x
        x = self.conv(x)
        return x

class decoder_unit(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(decoder_unit, self).__init__()
        self.block1 = nn.Sequential(
                nn.Conv2d(input_channel, input_channel* 2, kernel_size=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(input_channel* 2),

                nn.Conv2d(input_channel* 2, input_channel * 2, kernel_size=3, padding=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(input_channel* 2),

                nn.Conv2d(input_channel * 2, input_channel, kernel_size=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(input_channel),
            )

        self.conv = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(output_channel),
            )

    def forward(self, x):
        # residual = x
        x = self.block1(x) + x
        x = self.conv(x)
        return x

class diffusion_en(nn.Module):
    def __init__(self):
        super(diffusion_en, self).__init__()
        self.downsampling = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(16),

                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(16),
            )

        self.encoder = encoder_unit(16, 28)          

    def forward(self, x):
        x =  self.downsampling(x)
        transmit_init = self.encoder(x)
        return transmit_init

class diffusion_de(nn.Module):
    def __init__(self):
        super(diffusion_de, self).__init__()
        self.decoder = decoder_unit(28, 16)
        self.upsampling = nn.Sequential(
                nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.LeakyReLU(0.1, inplace=True),

                nn.Conv2d(16, 4, kernel_size=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(4),               
            )

        self.fc_mu = nn.Conv2d(4, 4, kernel_size=1, bias=False)
        self.fc_logvar = nn.Conv2d(5, 4, kernel_size=1, bias=False)

    def encode(self, x, snr_db, device):
        x = self.decoder(x)
        h = self.upsampling(x)

        B, _, W, H = h.shape
        tensor_snr = snr_db * torch.ones((B, 1, W, H)).to(device)
        hat_h = torch.cat((h,tensor_snr),1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(hat_h)
        return mu, logvar          

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, std

    def forward(self, x, snr_db, device):
        # print(x.shape)
        mu, logvar = self.encode(x, snr_db, device)
        z, std = self.reparameterize(mu, logvar)
        return z, mu, std

class model_loss(nn.Module): # 注意继承 nn.Module
    def __init__(self):
        super(model_loss, self).__init__()
        self.kl_divergence = nn.KLDivLoss()

    def forward(self, t_mu, t_std, s_mu, s_std):
        # transmit_s 的分布趋近于 transmit_t 的分布
        # KL(transmit_t||transmit_s) = transmit_t * log(transmit_t/transmit_s)
        # 高斯分布 = log(sqrt(transmit_s)/sqrt(transmit_t)) + pow(transmit_t) + pow[mean(transmit_t-transmit_s)]/(2*pow(transmit_s))
        # 均值方差均对batch/sample求, dim = batch

        loss = torch.log(s_std/t_std) + (t_std**2 + (t_mu - s_mu)**2) / (2 * s_std**2) - 0.5
        loss = torch.mean(loss)
        # torch.log(sigma_1 / sigma_0) + (sigma_0**2 + (mu_0 - mu_1)**2) / (2 * sigma_1**2) - 0.5
        return loss

class KL_loss(nn.Module): # 注意继承 nn.Module
    def __init__(self):
        super(KL_loss, self).__init__()
        self.kl_divergence = nn.KLDivLoss()

    def forward(self, s_mu, s_std):
        loss = - 0.5 * (1 + 2 * torch.log(s_std) - s_mu**2 - s_std**2 )
        loss = torch.mean(loss)
        # torch.log(sigma_1 / sigma_0) + (sigma_0**2 + (mu_0 - mu_1)**2) / (2 * sigma_1**2) - 0.5

        return loss

def train(opt, model, diffusion_encoder, diffusion_decoder, sampler, data, batch_size, train_loader, optimizer1, optimizer2, device):
    mseLoss = torch.nn.MSELoss(reduction='mean')
    for batch_idx, (init_image, file_name) in enumerate(train_loader):
        init_image = init_image.to(device)
        # print(init_image.shape)
        transmit_z = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    # 取 3dB 时的噪声分布
                    # encode (scaled latent) according to snr_db
                    snr_values = [-5,-3,-1,1,3,5,7,9,11,13] # 
                    # random.shuffle(snr_values)
                    for snr_dB in snr_values:  # 0,3,6,9,12

                        optimizer1.zero_grad()
                        optimizer2.zero_grad()

                        _snr_db = torch.tensor(snr_dB).to(device)
                        transmit_t, t_enc, hat_alphas_sqrt, grouptruth_std, noise, alpha = sampler.stochastic_encode_from_snr(transmit_z, _snr_db, batch_size)
                        print(f"target source t_enc is {t_enc} steps")

                        with torch.enable_grad():
                            transmit_s = diffusion_encoder(transmit_z)

                            transmit = form_signal_to_tensor(transmit_s, _snr_db, opt.CT)

                            print(f"target compress t_enc is {t_enc} steps")
                            transmit, predict_mu, predict_std = diffusion_decoder(transmit, _snr_db, device)

                            grouptruth_mu = hat_alphas_sqrt * transmit_z

                            # loss = opt.lamda2 * model_loss()(grouptruth_mu, grouptruth_std, predict_mu, predict_std)
                            loss = mseLoss(predict_mu, transmit_t) + opt.lamda1* KL_loss()(predict_mu, predict_std) + opt.lamda2 * model_loss()(grouptruth_mu, grouptruth_std, predict_mu, predict_std)
                            loss.backward()
                            optimizer1.step()
                            optimizer2.step()
                            if batch_idx % 10 == 0:
                                print('Train Epoch: [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                                    batch_idx * len(init_image), len(train_loader.dataset),
                                    100. * batch_idx / len(train_loader), loss.item()))

    return loss.item()/len(train_loader.dataset)

def test(opt, model, diffusion_encoder, diffusion_decoder, sampler, data, batch_size, val_loader, device):
    loss = 0
    mseLoss = torch.nn.MSELoss(reduction='mean')
    with torch.no_grad():
        for batch_idx, (init_image, file_name) in enumerate(val_loader):
            init_image = init_image.to(device)
            # print(init_image.shape)
            transmit_z = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
            sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
            precision_scope = autocast if opt.precision == "autocast" else nullcontext
            with precision_scope("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        for snr_dB in [10]:  # 0,3,6,9,12
                            # encode (scaled latent) according to snr_db
                            _snr_db = torch.tensor(snr_dB).to(device)
                            transmit_t, t_enc, hat_alphas_sqrt, grouptruth_std, noise, alpha = sampler.stochastic_encode_from_snr(transmit_z, _snr_db, batch_size, device)
                            print(f"target t_enc is {t_enc} steps")

                            transmit_s = diffusion_encoder(transmit_z)
                            transmit = form_signal_to_tensor(transmit_s, _snr_db, opt.CT)

                            print(f"target compress t_enc is {t_enc} steps")

                            transmit, predict_mu, predict_std = diffusion_decoder(transmit, snr_dB, device)
                            grouptruth_mu = hat_alphas_sqrt * transmit_z

                            # loss = opt.lamda2 * model_loss()(grouptruth_mu, grouptruth_std, predict_mu, predict_std)

                            loss = mseLoss(predict_mu, transmit_t) + opt.lamda1 * KL_loss()(predict_mu, predict_std) + opt.lamda2 * model_loss()(grouptruth_mu, grouptruth_std, predict_mu, predict_std)

    # return loss.item()/len(val_loader.dataset)

def evaluate_fid(opt, model, diffusion_encoder, diffusion_decoder, sampler, wm_encoder, data, batch_size, val_loader, device, sample_path):
    loss = 0
    base_count = 0
    mseLoss = torch.nn.MSELoss(reduction='mean')
    with torch.no_grad():
        for batch_idx, (init_image, file_name) in enumerate(val_loader):
            init_image = init_image.to(device)
            # print(init_image.shape)
            transmit_z = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
            sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
            precision_scope = autocast if opt.precision == "autocast" else nullcontext
            with precision_scope("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        snr_values = [1,3,5,7,9,11] # -5,-3,-1,1,3
                        random.shuffle(snr_values)
                        for snr_dB in snr_values:  # 0,3,6,9,12
                            # encode (scaled latent) according to snr_db
                            if not os.path.exists(sample_path+'/'+str(snr_dB)):
                                os.makedirs(sample_path+'/'+str(snr_dB))

                            _snr_db = torch.tensor(snr_dB).to(device)
                            transmit_t, t_enc, hat_alphas_sqrt, grouptruth_std, noise, _ = sampler.stochastic_encode_from_snr(transmit_z, _snr_db, batch_size, device)
                            print(f"target t_enc is {t_enc} steps")

                            transmit_s = diffusion_encoder(transmit_z)

                            try:
                                transmit = form_signal_to_tensor(transmit_s, _snr_db, opt.CT)
                            except:
                                print('error')
                                continue

                            transmit, predict_mu, predict_std = diffusion_decoder(transmit, snr_dB, device)

                            # # decode it
                            samples = sampler.decode(transmit, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc, )

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            nn = 0
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                # img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                # img.save(os.path.join(sample_path, file_name[nn]))
                                img.save(os.path.join(sample_path+'/'+str(snr_dB), f"{file_name[0]}-{base_count:05}.png"))
                                base_count += 1
                                nn += 1
                            all_samples.append(x_samples)

def save_checkpoint(state, loss, epoch, filepath):

    if not os.path.exists(filepath):
        os.makedirs(filepath)
    torch.save(state, os.path.join(filepath, str(epoch)+'_epoch_loss_'+str(loss)+'.pth'))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument("--epochs", type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')

    parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')

    parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset",
    )

    parser.add_argument(
        "--lr",
        type=float, 
        default=0.001,
        help='learning rate (default: 0.001)'
    )

    parser.add_argument(
        "--lamda1",
        type=float, 
        default = 0.1,
        help='MSE + lamda1 * KL'
    )

    parser.add_argument(
        "--lamda2",
        type=float, 
        default = 0.1,
        help='MSE + lamda1 * KL + lamda2 * model_loss'
    )

    parser.add_argument(
        "--CT",
        type=str, 
        default = 'awgn',
        help='channel type: awgn, fading, mimo'
    )

    parser.add_argument('--device', default='', 
                    help='device id (i.e. 0 or 0,1) or cpu')

    opt = parser.parse_args()
    print(opt)
    seed_everything(opt.seed)
    device = select_device(opt.device,apex=False)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)
    # sampler = DDPM(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # assert os.path.isfile(opt.init_img)

    # 实例化验证数据集
    if opt.dataset == 'bedroom':
        train_dataset = LSUNDataSet(root_path='LSUN/data/bedroom/bedroom_train_lmdb/', n_samples = opt.n_samples)
        val_dataset = LSUNDataSet(root_path='LSUN/data/bedroom/bedroom_val_lmdb/', n_samples = opt.n_samples)
        test_dataset = LSUNDataSet(root_path='LSUN/data/bedroom/bedroom_val_lmdb/', n_samples = 1)
    elif opt.dataset == 'church':
        # train_dataset = LSUNDataSet(root_path='LSUN/data/church/church_outdoor_train_lmdb_1000/', n_samples = opt.n_samples)
        train_dataset = LSUNDataSet(root_path='LSUN/data/church/church_outdoor_train_lmdb/', n_samples = opt.n_samples)
        val_dataset = LSUNDataSet(root_path='LSUN/data/church/church_outdoor_val_lmdb/', n_samples = opt.n_samples)
        test_dataset = LSUNDataSet(root_path='LSUN/data/church/church_outdoor_val_lmdb/', n_samples = opt.n_samples)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=test_dataset.collate_fn)

    # init_image = load_img(opt.init_img).to(device)
    # init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

    diffusion_encoder = diffusion_en().to(device)
    diffusion_decoder = diffusion_de().to(device)
    print(diffusion_encoder)
    diffusion_encoder.train()
    diffusion_decoder.train()

    if opt.refine:
        checkpoint = torch.load(opt.refine)
        diffusion_encoder.load_state_dict(checkpoint['en_state_dict'], strict=True)
        diffusion_decoder.load_state_dict(checkpoint['de_state_dict'], strict=True)

    # optimizer = torch.optim.Adam(diffusion_encoder.parameters(), lr=1e-3)
    for name, value in diffusion_encoder.named_parameters():
        group_name = name.split('.')
        value.requires_grad = True
        
    for name, value in diffusion_decoder.named_parameters():
        value.requires_grad = True

    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, diffusion_encoder.parameters()), lr=opt.lr)
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, diffusion_decoder.parameters()), lr=opt.lr)

    best_loss = 100

    # for epoch in range(opt.epochs):
    #     start_time = time.time()
    #     loss = train(opt, model, diffusion_encoder, diffusion_decoder, sampler, data, batch_size, train_loader, optimizer1, optimizer2, device)
    #     test(opt, model, diffusion_encoder, diffusion_decoder, sampler, data, batch_size, val_loader, device)
    #     during_time = time.time()-start_time
    #     print(during_time)

    #     is_best = loss < best_loss
    #     best_loss = min(loss, best_loss)       
    #     # if epoch % 10 == 0 or is_best:
    #     # if epoch % 10 == 0:
    #     save_checkpoint({
    #         'epoch': epoch,
    #         'en_state_dict': diffusion_encoder.state_dict(),
    #         'de_state_dict': diffusion_decoder.state_dict(),
    #         'en_optimizer': optimizer1.state_dict(),
    #         'de_optimizer': optimizer2.state_dict(),
    #     }, loss=loss ,epoch=epoch, filepath=opt.save)  

    diffusion_encoder.eval()
    diffusion_decoder.eval()

    evaluate_fid(opt, model, diffusion_encoder, diffusion_decoder, sampler, wm_encoder, data, batch_size, test_loader, device, sample_path)

    # print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")


if __name__ == "__main__":
    main()

# python scripts/train_submission.py --ckpt ../v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --init-img outputs/no_img_prompt/00000.png --strength 0.5 --ddim_step 400 --scale 0 --dataset church --n_samples 4 --save respond/ablation/church/  --outdir respond/ablation/church/   --lamda1 0.1 --lamda2 0.1  --lr 0.0001 --epochs 1000