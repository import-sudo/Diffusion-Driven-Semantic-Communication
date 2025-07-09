import torch
from torchvision.models import inception_v3
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from pytorch_fid import fid_score	
from PIL import Image
import os
import cv2
import numpy as np
# 生成的图像数据集（假设你有一个名为generated_data的数据集）
 
import lpips
from skimage.measure import compare_psnr, compare_ssim
import argparse, os

from torch_utils import *

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.images_path = os.listdir(files)
        self.transforms = transforms

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, i):
        path = self.files+'/'+self.images_path[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def is_value(generated_images_files, batch_size, device):
	# 加载预训练的Inception模型
	print(generated_images_files)
	inception_model = inception_v3(pretrained=True, transform_input=False).eval()
	inception_model.to(device)

	# 数据预处理
	transform = transforms.Compose([
	    transforms.Resize((512, 512)),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])

	dataset = ImagePathDataset(generated_images_files, transforms=transform)
	dataloader = torch.utils.data.DataLoader(dataset,
	                                         batch_size=batch_size,
	                                         shuffle=False,
	                                         drop_last=False,
	                                         num_workers=1)
	# 计算预测分数
	predictions = []
	for images in dataloader:
	    images = images.to(device)
	    preds = inception_model(images)
	    predictions.append(torch.softmax(preds, dim=1))

	# 将预测结果转换为张量
	predictions = torch.cat(predictions, dim=0)

	# 计算真实性分数和多样性分数
	realism_score = torch.mean(torch.max(predictions, dim=1)[0].log())
	diversity_score = torch.exp(torch.mean(torch.sum(predictions * torch.log(predictions), dim=1)))

	# 计算最终的Inception Score
	inception_score = torch.exp(realism_score) * diversity_score
	print("Inception Score:", inception_score.item())
	return inception_score

def fid_value(real_images_folder,generated_images_folder, batch_size, device, dims):
	# 计算FID距离值
	fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder], batch_size, device, dims)
	print('FID value:', fid_value)
	return fid_value

def eval_clip(original_dir, generated_dir, device):
	import clip

	model, preprocess = clip.load("ViT-B-32.pt", device=device)

	with torch.no_grad():
		mean_similarity_score = 0
		count_image = 0 
		for name1 in os.listdir(original_dir):	
			image1 = preprocess(Image.open(original_dir+name1)).unsqueeze(0).to(device)
			image1_features = model.encode_image(image1)
			similarity_score = 0
			count = 0
			for name2 in os.listdir(generated_dir):
				if name2.split('-')[0] == name1:
					image2 = preprocess(Image.open(generated_dir+name2)).unsqueeze(0).to(device)
					image2_features = model.encode_image(image2)
					count +=1 

					similarity_score += torch.nn.functional.cosine_similarity(image1_features, image2_features)
			if count == 0:
				continue			
			count_image += 1
			similarity_score /= count
			mean_similarity_score += similarity_score

		mean_similarity_score /= count_image
	print(mean_similarity_score)
	return mean_similarity_score

def eval_lpips(original_dir, generated_dir, device, model='alex'):
	if model == 'alex':
		lpips_model = lpips.LPIPS(net='alex').to(device) # best forward scores
	else:
		lpips_model = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

	with torch.no_grad():
		mean_score = 0
		count_image = 0 
		for name1 in os.listdir(original_dir):	
			img1 = lpips.load_image(original_dir+name1)
			img1 = cv2.resize(img1, (512,512))
			image1_features = lpips.im2tensor(img1).to(device)
			score = 0
			count = 0
			for name2 in os.listdir(generated_dir):
				if name2.split('-')[0] == name1:
					image2_features = lpips.im2tensor(lpips.load_image(generated_dir+name2)).to(device)
					count +=1 
					score += lpips_model(image1_features, image2_features)

			if count == 0:
				continue
			count_image += 1
			score /= count
			mean_score += score

		mean_score /= count_image

	return mean_score

def eval_psnr_ssim(original_dir, generated_dir):
	with torch.no_grad():
		mean_psnr = 0
		mean_ssim = 0
		count_image = 0 
		for name1 in os.listdir(original_dir):
			image1 = Image.open(original_dir+name1).resize((512, 512), resample=Image.LANCZOS) #>>C,H,R
			# image1 = image1.resize((512, 512), resample=Image.LANCZOS)
			# print(np.asarray(image1).shape)
			psnr = 0
			ssim = 0
			count = 0
			for name2 in os.listdir(generated_dir):
				if name2.split('-')[0] == name1:
					image2 = Image.open(generated_dir+name2).resize((512, 512), resample=Image.LANCZOS)
					count +=1 
					mssim = [compare_ssim(np.asarray(image1)[:, :, i], np.asarray(image2)[:, :, i], data_range=255) for i in range(3)]
					ssim += np.mean(mssim)
					mpsnr = [compare_psnr(np.asarray(image1)[:, :, i], np.asarray(image2)[:, :, i], data_range=255) for i in range(3)]
					psnr += np.mean(mpsnr)

					# psnr += compare_psnr(np.asarray(image1).reshape(-1,512, 512), np.asarray(image2).reshape(-1,512, 512), data_range=255)
					# ssim += compare_ssim(np.asarray(image1).reshape(-1,512, 512), np.asarray(image2).reshape(-1,512, 512), data_range=255)
			if count == 0:
				continue

			psnr /= count
			ssim /= count
			mean_psnr += psnr
			mean_ssim += ssim
			count_image += 1

		mean_psnr /= count_image
		mean_ssim /= count_image

	return mean_psnr, mean_ssim


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='', 
                help='device id (i.e. 0 or 0,1) or cpu')

opt = parser.parse_args()
print(opt)
device = select_device(opt.device,apex=False)

batch_size = 32
dims = 2048
print(device)

original_dir = 'LSUN/data/church/church_outdoor_val_lmdb/'
generated_dir = 'respond/mimo_our/church/samples/'


snr_list = []
ssim_list = []
lpips_list =[]
FID_list = []
clip_list = []
with torch.no_grad(): 
	with open('respond/results.txt', 'a+') as f:
		for snr in ['1','3','5','7','9','11']:  

			target_dir = generated_dir+snr+'/'
			print(target_dir)
			mean_psnr, mean_ssim = eval_psnr_ssim(original_dir, target_dir)
			lpips_score = eval_lpips(original_dir, target_dir, device, model='alex')
			clip_sorce = eval_clip(original_dir, target_dir, device)
			FID = fid_value(original_dir, target_dir,batch_size, device, dims)
			print(mean_psnr, mean_ssim, lpips_score, FID, clip_sorce)
			f.write(f"target_dir: {target_dir}, PSNR: {mean_psnr}, SSIM: {mean_ssim}, LPIPS: {lpips_score.item()}, FID: {FID}, CLIP Score: {clip_sorce.item()}\n")

			snr_list.append(round(mean_psnr, 4))
			ssim_list.append(round(mean_ssim, 4))
			lpips_list.append(round(lpips_score.item(), 4))
			FID_list.append(round(FID, 4))
			clip_list.append(round(clip_sorce.item(), 4))

	print(snr_list)
	print(ssim_list)
	print(lpips_list)
	print(FID_list)
	print(clip_list)
