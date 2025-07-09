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
from torch_utils import *

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
        # img = img.resize((768, 768), resample=Image.LANCZOS)
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
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    # model = DataParallel(model)

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

    assert os.path.isfile(opt.init_img)

    img_size = 512
    data_transform = {
        "val": transforms.Compose([transforms.Resize([int(img_size),int(img_size)]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # # 实例化验证数据集
    # val_dataset = ImageNetDataSet(images_path='/gpu02/dataset_public/ILSVRC/validation_label.txt',
    #                         root_path='/gpu02/dataset_public/ILSVRC/Data/CLS-LOC/val/',
    #                         transform=data_transform["val"])

    # 'LSUN/data/church/church_outdoor_val_lmdb/'
    # 'LSUN/data/bedroom/bedroom_val_lmdb/'

    if opt.dataset == 'bedroom':
        val_dataset = LSUNDataSet(root_path='LSUN/data/bedroom/bedroom_val_lmdb/', n_samples = opt.n_samples)
    elif opt.dataset == 'church':
        val_dataset = LSUNDataSet(root_path='LSUN/data/church/church_outdoor_val_lmdb/', n_samples = opt.n_samples)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # init_image = load_img(opt.init_img).to(device)
    # init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    base_count = 0
    all_signal_pow = 0
    all_noise_pow = 0
    for batch_idx, (init_image, file_name) in enumerate(val_loader):
        init_image = init_image.to(device)
        print(init_image.shape)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

        assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(opt.strength * opt.ddim_steps)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            for channel_type in ['fading']: #'fading', 'mimo', 'awgn'
                                for snr_dB in [0,3,6,9,12]:
                                # for snr_dB in [0,3,6,9,12]:
                                    if not os.path.exists(sample_path+'/'+ channel_type +'_no_normalize/'+str(snr_dB)):
                                        os.makedirs(sample_path+'/'+ channel_type +'_no_normalize/'+str(snr_dB))

                                    _snr_db = torch.tensor(snr_dB).to(device)

                                    z_enc, t_enc = sampler.stochastic_encode_according_snr(init_latent, _snr_db, batch_size, channel_type)
                                    print(f"target t_enc is {t_enc} steps")

                                    # # decode it
                                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                             unconditional_conditioning=uc, )

                                    # samples = init_latent

                                    x_samples = model.decode_first_stage(samples)
                                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                    nn = 0
                                    for x_sample in x_samples:
                                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                        img = Image.fromarray(x_sample.astype(np.uint8))
                                        img = put_watermark(img, wm_encoder)
                                        # img.save(os.path.join(sample_path, f"{file_name[0]}-{base_count:05}.png"))
                                        # img.save(os.path.join(sample_path, file_name[nn]))
                                        img.save(os.path.join(sample_path+'/'+ channel_type +'_no_normalize/'+str(snr_dB), f"{file_name[0]}-{base_count:05}.png"))
                                        base_count += 1
                                        nn += 1
                                    all_samples.append(x_samples)

                    # additionally, save as grid
                    # grid = torch.stack(all_samples, 0)
                    # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    # grid = make_grid(grid, nrow=n_rows)

                    # # to image
                    # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    # grid = Image.fromarray(grid.astype(np.uint8))
                    # grid = put_watermark(grid, wm_encoder)
                    # grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    # grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")


if __name__ == "__main__":
    main()


    ### python scripts/img2img_with_wireless_channel.py --ckpt v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml  --strength 0.5 --ddim_step 400 --scale 0 --dataset bedroom --n_samples 2 --outdir bedroom/