import numpy as np
import cv2
import os
from skimage.io import imsave
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import random
import h5py
import torch.utils.data as Data
from skimage.io import imread
from pytorch_msssim import SSIM
from SRutils.Evaluator_deepseeker import Runevaluator
from nets.Ufuser import Ufuser
from nets.Ufuserplus import Ufuserplus
from typing import Literal, Tuple
from torch import autograd as autograd

ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)
def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype(np.float32)
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def rgb2y(img):
    # 标准的灰度值转换公式
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    imsave(os.path.join(savepath, "{}.png".format(imagename)), image)


class loss_fusion(nn.Module):
    def __init__(self,coeff_int=1.0,coeff_grad=10.0,coeff_grad2=1.0,coeff_strcut=0.1):
        super(loss_fusion, self).__init__()
        self.coeff_int=coeff_int
        self.coeff_grad=coeff_grad
        self.coeff_grad2=coeff_grad2
        self.coeff_strcut=coeff_strcut
        self.sobelconv=Sobelxy(device='cuda')
    def makegrad(self, image):
        shape = image.shape
        height = shape[2]
        width = shape[3]
        batch = shape[0]
        grad_2 = kornia.filters.SpatialGradient()(image)
        grad_2x = grad_2[:, :, 0:1, :, :].reshape(batch, 1, height, width)
        grad_2y = grad_2[:, :, 1:2, :, :].reshape(batch, 1, height, width)
        # grad_22 = torch.sqrt(grad_2x ** 2 + grad_2y ** 2 + 1e-6)  # Add small constant
        # grad_22 = grad_22/torch.max(grad_22)  # Normalize
        grad_22x = torch.exp(grad_2x)/(torch.exp(grad_2x)+torch.exp(grad_2y))
        grad_22y = torch.exp(grad_2y)/(torch.exp(grad_2x)+torch.exp(grad_2y))
        grad_22 = torch.sqrt((grad_2x*grad_22x)** 2 + (grad_2y*grad_22y)** 2+ 1e-6)
        return grad_22
    
    def structural_loss(self,fused, visible):
        loss= 1 - ssim_loss(fused, visible)
        return loss
    def forward(self,pre,target):
        # 后面跟的是原图，前面是拆出来的图
        # 这里加了可见光图像补强
        # 将特征图作为一种损失
        loss_int=F.l1_loss(pre,target)
        loss_grad1=F.l1_loss(kornia.filters.SpatialGradient()(pre),kornia.filters.SpatialGradient()(target))
        loss_grad2 = F.l1_loss(self.sobelconv(pre),self.sobelconv(target))
        structloss = self.structural_loss(pre,target)
        grad_to = self.coeff_grad*loss_grad1 + self.coeff_grad2*loss_grad2
        loss_total= self.coeff_int*loss_int+ self.coeff_grad2*loss_grad2 + self.coeff_strcut * structloss
        
        return loss_total
class loss_fusion_f(nn.Module):
    def __init__(self,coeff_int=1.0,coeff_grad=1.0,coeff_grad1=1.0,coeff_grad2=10.0,coeff_strcut=0.1):
        super(loss_fusion_f, self).__init__()
        self.coeff_int=coeff_int
        self.coeff_grad=coeff_grad
        self.coeff_grad1=coeff_grad1
        self.coeff_grad2=coeff_grad2
        self.coeff_strcut=coeff_strcut
        self.sobelconv=Sobelxy(device='cuda')
    def structural_loss(self,fused, visible):
        loss= 1 - ssim_loss(fused, visible)
        return loss
    def region_mean(x, mask):
        return (x*mask).sum() / mask.sum().clamp_min(1)
    def smooth_max(self, a, b, beta=10.0):
        return (torch.log(torch.exp(beta * a) + torch.exp(beta * b)) / beta)
    def forward(self,vi,ir,f):
        loss_int=F.l1_loss(f,torch.max(vi, ir))
        loss_grad1=F.l1_loss(self.sobelconv(f),torch.max(self.sobelconv(vi), self.sobelconv(ir)))
        loss_grad2=F.l1_loss(kornia.filters.SpatialGradient()(f),torch.max(kornia.filters.SpatialGradient()(vi),kornia.filters.SpatialGradient()(ir)))
        loss_grad_to = self.coeff_grad1*loss_grad1 + self.coeff_grad2*loss_grad2
        structloss = self.structural_loss(vi,f)+self.structural_loss(ir,f)
        # 这里改一下损失
        loss_total=self.coeff_int*loss_int+ self.coeff_grad1*loss_grad1
        return loss_total


class loss_fusion_m(nn.Module):
    def __init__(self,coeff_int=1.0,coeff_grad=1.0,coeff_grad1=1.0,coeff_grad2=10.0,coeff_strcut=0.1):
        super(loss_fusion_m, self).__init__()
        self.coeff_int=coeff_int
        self.coeff_grad=coeff_grad
        self.coeff_grad1=coeff_grad1
        self.coeff_grad2=coeff_grad2
        self.coeff_strcut=coeff_strcut
        self.sobelconv=Sobelxy(device='cuda')
    def structural_loss(self,fused, visible):
        loss= 1 - ssim_loss(fused, visible)
        return loss
    def forward(self,vi,ir,f):
        loss_int=F.l1_loss(f,torch.max(vi, ir))
        loss_grad1=F.l1_loss(self.sobelconv(f),torch.max(self.sobelconv(vi), self.sobelconv(ir)))
        loss_grad2=F.l1_loss(kornia.filters.SpatialGradient()(f),torch.max(kornia.filters.SpatialGradient()(vi),kornia.filters.SpatialGradient()(ir)))
        loss_grad_to =  self.coeff_grad1*loss_grad1 + self.coeff_grad2*loss_grad2
        structloss = self.structural_loss(vi,f)+self.structural_loss(ir,f)
        # 这里改一下损失
        loss_total=self.coeff_int*loss_int+ + self.coeff_grad*loss_grad1 + self.coeff_strcut * structloss
        return loss_total

# 正则化的tvloss
def tv_loss(x):
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w

class Sobelxy(nn.Module):
    def __init__(self,device='cuda'):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        grad = torch.sqrt(sobelx*sobelx + sobely*sobely + 1e-6)
        return grad
        
class Fusionloss(nn.Module):
    def __init__(self,coeff_int=1,coeff_grad=10,in_max=True, device='cuda'):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy(device=device)
        self.coeff_int=coeff_int
        self.coeff_grad=coeff_grad
        self.in_max=in_max
    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        if self.in_max:
            x_in_max=torch.max(image_y,image_ir)
        else:
            x_in_max=(image_y+image_ir)/2.0
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=self.coeff_int*loss_in+self.coeff_grad*loss_grad
        return loss_total,loss_in,loss_grad
    
class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        h5f.close()
        return torch.Tensor(IR), torch.Tensor(VIS), index
    
class H5Dataset_AiAv(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['input_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['input_patchs'][key])
        VIS = np.array(h5f['target_patchs'][key])
        h5f.close()
        return torch.Tensor(IR), torch.Tensor(VIS),index

def Result_Save(epoch, model_path, text_path, name, data_type):
    if data_type == 'MSRS':
        path_ir=r"../MSRS-main/test/Inf"
        path_vi=r"../MSRS-main/test/vi"
    if data_type == 'M3FD':
        path_ir=r"../M3FD/IR"
        path_vi=r"../M3FD/VI"
    if data_type == 'Harvard':
        path_ir=r"../Harvard/MRI"
        path_vi=r"../Harvard/T"
    if data_type == 'FMB':
        path_ir=r"../FMB/ir"
        path_vi=r"../FMB/vi"
    path_model=model_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model=Ufuserplus().to(device)
    model.load_state_dict(torch.load(path_model))
    model.eval()
    target = name
    path_save=os.path.join(r"test_result" , data_type + "_" + target, str(epoch))

    with torch.no_grad():
        for imgname in os.listdir(path_ir):
        
            IR = image_read_cv2(os.path.join(path_ir, imgname), 'GRAY')[np.newaxis,np.newaxis,...]/255
            VI = image_read_cv2(os.path.join(path_vi, imgname), 'GRAY')[np.newaxis,np.newaxis,...]/255

            h, w = IR.shape[2:]
            h1 = h - h % 32
            w1 = w - w % 32
            h2 = h % 32
            w2 = w % 32

            if h1==h and w1==w: 
                ir = ((torch.FloatTensor(IR))).to(device)
                vi = ((torch.FloatTensor(VI))).to(device)
                data_Fuse=model(vi,ir)
                data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
                fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
                fused_image = fused_image.astype(np.uint8)
                img_save(fused_image, imgname.split(sep='.')[0], path_save)
            else:
                # Upper left 
                fused_temp=np.zeros((h,w),dtype=np.float32)
                ir_temp = ((torch.FloatTensor(IR))[:,:,:h1,:w1]).to(device)
                vi_temp = ((torch.FloatTensor(VI))[:,:,:h1,:w1]).to(device)
                data_Fuse=model(vi_temp,ir_temp)
                fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
                fused_temp[:h1,:w1]=fused_image

                # upper right
                if w1!=w:
                    ir_temp = ((torch.FloatTensor(IR))[:,:,:h1,-w1:]).to(device)
                    vi_temp = ((torch.FloatTensor(VI))[:,:,:h1,-w1:]).to(device)
                    data_Fuse=model(vi_temp,ir_temp)
                    fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
                    fused_temp[:h1,-w2:]=fused_image[:,-w2:]

                # lower left
                if h1!=h:    
                    ir_temp = ((torch.FloatTensor(IR))[:,:,-h1:,:w1]).to(device)
                    vi_temp = ((torch.FloatTensor(VI))[:,:,-h1:,:w1]).to(device)
                    data_Fuse=model(vi_temp,ir_temp)
                    fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
                    fused_temp[-h2:,:w1]=fused_image[-h2:,:]

                
                # lower right
                if h1!=h and w1!=w:
                    ir_temp = ((torch.FloatTensor(IR))[:,:,-h1:,-w1:]).to(device)
                    vi_temp = ((torch.FloatTensor(VI))[:,:,-h1:,-w1:]).to(device)
                    data_Fuse=model(vi_temp,ir_temp)
                    fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
                    fused_temp[-h2:,-w2:]=fused_image[-h2:,-w2:]

                fused_temp=(fused_temp-np.min(fused_temp))/(np.max(fused_temp)-np.min(fused_temp))
                fused_temp=((fused_temp)*255).astype(np.uint8)
                img_save(fused_temp, imgname.split(sep='.')[0], path_save) 

    EN,SD,SF,AG,SCD,VIFF,MI,SSIM,Qbaf,count = Runevaluator(target,path_save,epoch,data_type,path_ir,path_vi)
    EN,SD,SF,AG,SCD,VIFF,MI,SSIM,Qbaf = EN.item(),SD.item(),SF.item(),AG.item(),SCD.item(),VIFF.item(),MI.item(),SSIM.item(),Qbaf.item()
    with open(text_path, 'a', encoding='utf-8') as file:
        file.write(f"第{epoch}轮的数据:")
        file.write(f"{EN/count:.2f}\t{SD/count:.2f}\t{SF/count:.2f}\t"
                f"{AG/count:.2f}\t{SCD/count:.2f}\t{VIFF/count:.2f}\t{MI/count:.2f}\t{Qbaf/count:.2f}\t{SSIM/count:.2f}\n")        

# 生成对抗损失
class UNGANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(UNGANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss * self.loss_weight

class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


class MultiScaleGANLoss(GANLoss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(MultiScaleGANLoss, self).__init__(gan_type, real_label_val, fake_label_val, loss_weight)

    def forward(self, input, target_is_real, is_disc=False):
        """
        The input is a list of tensors, or a list of (a list of tensors)
        """
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    # Only compute GAN loss for the last layer
                    # in case of multiscale feature matching
                    pred_i = pred_i[-1]
                # Safe operation: 0-dim tensor calling self.mean() does nothing
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Reference: Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(outputs=real_pred.mean(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty
