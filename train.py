from __future__ import print_function
import argparse
import os
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM, ssim
from utils import save_img, VGGPerceptualLoss
from network1 import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from model_with_eca import Restormer
import kornia
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='Phaseformer')
parser.add_argument('--dataset', required=False, default='./uw_data/', help='dataset folder')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--finetune', default=False, help='to finetune')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=400, help='# of epochs at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=400, help='# of epochs to decay learning rate')
parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy')
parser.add_argument('--lr_decay_iters', type=int, default=500, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader')
parser.add_argument('--seed', type=int, default=123, help='random seed')
opt = parser.parse_args()
print(opt)


# Custom dataset class to load images from multiple directories
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dirs, reference_dirs, transform=None, transformH=None):
        """
        Args:
            input_dirs (list): List of directories containing input images.
            reference_dirs (list): List of directories containing reference images.
            transform (callable, optional): Optional transform to be applied to input images.
            reference_transform (callable, optional): Optional transform to be applied to reference images.
        """
        self.input_dirs = input_dirs
        self.reference_dirs = reference_dirs
        self.transform = transform
        self.transformH = transformH
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        # Gather image paths from multiple directories
        self.input_images = []
        self.reference_images = []
        for input_dir, reference_dir in zip(input_dirs, reference_dirs):
            input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)])
            reference_files = sorted([f for f in os.listdir(reference_dir) if f.lower().endswith(valid_extensions)])
            # Ensure filenames match, skip if not
            for file in input_files:
                if file in reference_files:
                    self.input_images.append(os.path.join(input_dir, file))
                    self.reference_images.append(os.path.join(reference_dir, file))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # Load images and apply transformations
        input_image = Image.open(self.input_images[idx]).convert("RGB")
        reference_image = Image.open(self.reference_images[idx]).convert("RGB")
        input_image = self.transform(input_image)
        reference_imageL = self.transform(reference_image)
        reference_imageH = self.transformH(reference_image)
        return input_image, reference_imageL, reference_imageH, idx


# Data loading functions for multiple directories
def get_dataset(root_dirs):
    """
    Args:
        root_dirs (dict): Dictionary containing dataset folder paths with 'input' and 'reference' keys.
                          Example: {'input': ['/path/to/uw_data', '/path/to/LUSI'],
                                    'reference': ['/path/to/ref_data1', '/path/to/ref_data2']}
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transformH = transforms.Compose([
        transforms.Resize((256 * 2, 256 * 2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_dirs = root_dirs['input']
    reference_dirs = root_dirs['reference']
    return CustomImageDataset(input_dirs, reference_dirs, transform, transformH)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda:0" if opt.cuda else "cpu")

if opt.finetune:
    G_path = 'Provide the path to the checkpoint you wish to continue from'
    net_g = torch.load(G_path).to(device)
else:
    net_g = Restormer().to(device)

print(f'Trainable parameters: {count_parameters(net_g)}')

print('===> Loading datasets')
root_dirs_train = {
    'input': ['./uw_data/train/a'],
    'reference': ['./uw_data/train/b']
}
dataset_train = get_dataset(root_dirs_train)
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
print(len(data_loader_train))

root_dirs_test = {
    'input': ['./uw_data/test/a'],
    'reference': ['./uw_data/test/b']
}
dataset_test = get_dataset(root_dirs_test)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=True)

print('===> Building models')


class Gradient_Loss(nn.Module):
    def __init__(self):
        super(Gradient_Loss, self).__init__()
        kernel_g = [[[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                    [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                    [[0, 1, 0], [1, -4, 1], [0, 1, 0]]]
        kernel_g = torch.FloatTensor(kernel_g).unsqueeze(0).permute(1, 0, 2, 3)
        self.weight_g = nn.Parameter(data=kernel_g, requires_grad=False)

    def forward(self, x, xx):
        grad = 0
        y = x
        yy = xx
        gradient_x = F.conv2d(y, self.weight_g, groups=3)
        gradient_xx = F.conv2d(yy, self.weight_g, groups=3)
        l = nn.L1Loss()
        a = l(gradient_x, gradient_xx)
        grad = grad + a
        return grad


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class WeightedLoss(nn.Module):
    """ Weighted Loss for upsampled and downsampled images loss"""

    def __init__(self, num_weights):
        super(WeightedLoss, self).__init__()
        self.num_weights = num_weights
        self.weights = nn.Parameter(torch.rand(1, num_weights))
        self.softmax_l = nn.Softmax(dim=1)

    def forward(self, *argv):
        loss = 0
        weights = self.softmax_l(self.weights)
        for idx, arg in enumerate(argv):
            loss += arg * weights[0, idx]
        return loss


# Define dynamic weights for losses
N = 4  # number of losses
Weighted_Loss4 = WeightedLoss(N).to(device)
Weighted_Loss2 = WeightedLoss(2).to(device)

# Define losses
Gradient_Loss = Gradient_Loss().to(device)
L_per = VGGPerceptualLoss().to(device)
MS_SSIM_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3).to(device)
Charbonnier_loss = nn.SmoothL1Loss().to(device)

optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)

output_dir_train = './images_train'
if not os.path.exists(output_dir_train):
    os.makedirs(output_dir_train)


# ---------------------------------------------------------------------
# Y-channel PSNR utilities
# ---------------------------------------------------------------------
def denorm(x):
    """Reverse Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)): [-1,1] -> [0,1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)


def rgb_to_y(x):
    """x: (B,3,H,W) in [0,1] range -> Y channel (B,1,H,W), BT.601 luma"""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def calculate_y_psnr(pred, target):
    """pred, target: (B,3,H,W) tensors in [-1,1] range"""
    pred = denorm(pred)
    target = denorm(target)
    pred_y = rgb_to_y(pred)
    target_y = rgb_to_y(target)
    mse = F.mse_loss(pred_y, target_y)
    if mse.item() == 0:
        return 100.0
    return 10 * log10(1.0 / mse.item())


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    net_g.train()

    train_pbar = tqdm(data_loader_train, desc=f"Epoch [{epoch}] Train", unit="batch")
    for iteration, batch in enumerate(train_pbar, 1):
        rgb, tarL, tarH, indx = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3]

        fake_b, fake_b1 = net_g(rgb)

        optimizer_g.zero_grad()

        loss_g_L = Weighted_Loss4(Charbonnier_loss(tarL, fake_b), L_per(fake_b, tarL),
                                   Gradient_Loss(fake_b, tarL), MS_SSIM_loss(fake_b, tarL))
        loss_g_H = Weighted_Loss4(Charbonnier_loss(tarH, fake_b1), L_per(fake_b1, tarH),
                                   Gradient_Loss(fake_b1, tarH), MS_SSIM_loss(fake_b1, tarH))
        loss_g = Weighted_Loss2(loss_g_L, loss_g_H)

        loss_g.backward()
        optimizer_g.step()

        if iteration % 100 == 0:
            out_image = torch.cat((rgb, fake_b, tarL), 3)
            save_img(out_image[0].detach().cpu(), f'{output_dir_train}/{iteration}.png')

        train_pbar.set_postfix(loss_g=f"{loss_g.item():.4f}")

    train_pbar.close()

    update_learning_rate(net_g_scheduler, optimizer_g)

    # -------------------------------------------------------------
    # Testing / Validation loop with average Y-channel PSNR
    # -------------------------------------------------------------
    output_dir = './images_test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    net_g.eval()
    avg_ypsnr = 0.0
    with torch.no_grad():
        test_pbar = tqdm(data_loader_test, desc=f"Epoch [{epoch}] Test", unit="batch")
        for test_iter, batch in enumerate(test_pbar, 1):
            rgb_input, target, targetH, idx = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3]

            prediction = net_g(rgb_input)[0]

            y_psnr = calculate_y_psnr(prediction, target)
            avg_ypsnr += y_psnr

            out = torch.cat((rgb_input, prediction, target), 3)
            filename = f'{output_dir}/{idx[0]}.png'  # Append the .png extension
            save_img(out[0].detach().cpu(), filename)

            test_pbar.set_postfix(y_psnr=f"{y_psnr:.4f}")

        test_pbar.close()

    avg_ypsnr = avg_ypsnr / len(data_loader_test)
    print(f"===> Epoch[{epoch}] Avg. Y-PSNR: {avg_ypsnr:.4f} dB")

    # -------------------------------------------------------------
    # Save checkpoint with avg Y-PSNR in filename
    # -------------------------------------------------------------
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))

    net_g_model_out_path = f"checkpoint/{opt.dataset}/netG_model_epoch_{epoch}_psnr_{avg_ypsnr:.4f}.pth"
    torch.save(net_g, net_g_model_out_path)
    print(f"Checkpoint saved at {net_g_model_out_path}")