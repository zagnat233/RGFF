import numpy as np
import cv2
import sklearn.metrics as skm
import math
from skimage.metrics import structural_similarity as ssim
import os
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  # 读出来不完全是整数，若需要整数则要round
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

class Evaluator():
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None):  # 检查输入
        if imgA is None:
            assert type(imgF) == torch.Tensor, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == torch.Tensor, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):  # entropy
        cls.input_check(img)
        a = torch.round(img).flatten().to(device)
        h = torch.bincount(a.long()) / a.shape[0]
        return -torch.sum(h * torch.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return torch.std(img)

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return torch.sqrt(torch.mean((img[:, 1:] - img[:, :-1]) ** 2) + torch.mean((img[1:, :] - img[:-1, :]) ** 2))

    @classmethod
    def AG(cls, img):  # Average gradient
        cls.input_check(img)
        Gx, Gy = torch.zeros_like(img), torch.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return torch.mean(torch.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return skm.mutual_info_score(image_F.flatten().cpu().numpy(), image_A.flatten().cpu().numpy()) + skm.mutual_info_score(image_F.flatten().cpu().numpy(),
                                                                                                   image_B.flatten().cpu().numpy())

    @classmethod
    def MSE(cls, image_F, image_A, image_B):  # MSE
        cls.input_check(image_F, image_A, image_B)
        return (torch.mean((image_A - image_F) ** 2) + torch.mean((image_B - image_F) ** 2)) / 2

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = torch.sum((image_A - torch.mean(image_A)) * (image_F - torch.mean(image_F))) / torch.sqrt(
            (torch.sum((image_A - torch.mean(image_A)) ** 2)) * (torch.sum((image_F - torch.mean(image_F)) ** 2)))
        rBF = torch.sum((image_B - torch.mean(image_B)) * (image_F - torch.mean(image_F))) / torch.sqrt(
            (torch.sum((image_B - torch.mean(image_B)) ** 2)) * (torch.sum((image_F - torch.mean(image_F)) ** 2)))
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * torch.log10(torch.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))

    @classmethod
    def SCD(cls, image_F, image_A, image_B): # The sum of the correlations of differences
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = torch.sum((image_A - torch.mean(image_A)) * (imgF_B - torch.mean(imgF_B))) / torch.sqrt(
            (torch.sum((image_A - torch.mean(image_A)) ** 2)) * (torch.sum((imgF_B - torch.mean(imgF_B)) ** 2)))
        corr2 = torch.sum((image_B - torch.mean(image_B)) * (imgF_A - torch.mean(imgF_A))) / torch.sqrt(
            (torch.sum((image_B - torch.mean(image_B)) ** 2)) * (torch.sum((imgF_A - torch.mean(imgF_A)) ** 2)))
        return corr1 + corr2

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F)+cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls, ref, dist):  # viff of a pair of pictures
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]  # 使用 NumPy 的 ogrid
            y = torch.from_numpy(y).float().to(device)  # 转换为 PyTorch 张量
            x = torch.from_numpy(x).float().to(device)  # 转换为 PyTorch 张量
            h = torch.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < torch.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = torch.conv2d(ref.unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze()
                dist = torch.conv2d(dist.unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze()
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = torch.conv2d(ref.unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze()
            mu2 = torch.conv2d(dist.unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze()
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = torch.conv2d((ref * ref).unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze() - mu1_sq
            sigma2_sq = torch.conv2d((dist * dist).unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze() - mu2_sq
            sigma12 = torch.conv2d((ref * dist).unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze() - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += torch.sum(torch.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if torch.isnan(vifp):
            return 1.0
        else:
            return vifp
        
    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        # 计算QABF
        deno = torch.sum(gA + gB)
        nume = torch.sum(torch.multiply(QAF, gA) + torch.multiply(QBF, gB))
        return nume / deno

    @classmethod
    def Qabf_getArray(cls, img):
        # Sobel Operator Sobel算子
        h1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).to(device)
        h2 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32).to(device)
        h3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(device)

        # 扩展输入和卷积核的维度
        img = img.unsqueeze(0).unsqueeze(0)  # 从 (H, W) 扩展为 (1, 1, H, W)
        h1 = h1.unsqueeze(0).unsqueeze(0)    # 从 (3, 3) 扩展为 (1, 1, 3, 3)
        h2 = h2.unsqueeze(0).unsqueeze(0)    # 从 (3, 3) 扩展为 (1, 1, 3, 3)
        h3 = h3.unsqueeze(0).unsqueeze(0)    # 从 (3, 3) 扩展为 (1, 1, 3, 3)

        # 计算梯度
        SAx = torch.conv2d(img, h3, padding=1).squeeze()  # padding=1 实现 'same' 效果
        SAy = torch.conv2d(img, h1, padding=1).squeeze()  # padding=1 实现 'same' 效果

        # 计算梯度和角度
        gA = torch.sqrt(SAx ** 2 + SAy ** 2)
        aA = torch.zeros_like(img.squeeze())
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0] = torch.atan(SAy[SAx != 0] / SAx[SAx != 0])

        return gA, aA
    @classmethod
    def Qabf_getQabf(cls,aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF,AAF,QgAF,QaAF,QAF = torch.zeros_like(aA),torch.zeros_like(aA),torch.zeros_like(aA),torch.zeros_like(aA),torch.zeros_like(aA)
        GAF[gA>gF]=gF[gA>gF]/gA[gA>gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA <gF] = gA[gA<gF]/gF[gA<gF]
        AAF = 1 - torch.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))
        QAF = QgAF* QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return ssim(image_F.cpu().numpy(),image_A.cpu().numpy(), data_range=255)+ssim(image_F.cpu().numpy(),image_B.cpu().numpy(), data_range=255)


# def VIFF1(image_F, image_A, image_B):
#     refA = image_A
#     refB = image_B
#     dist = image_F

#     sigma_nsq = 2
#     eps = 1e-10
#     numA = 0.0
#     denA = 0.0
#     numB = 0.0
#     denB = 0.0
#     for scale in range(1, 5):
#         N = 2 ** (4 - scale + 1) + 1
#         sd = N / 5.0

#         # Create a Gaussian kernel as MATLAB's
#         m, n = [(ss - 1.) / 2. for ss in (N, N)]
#         y, x = np.ogrid[-m:m + 1, -n:n + 1]  # 使用 NumPy 的 ogrid
#         y = torch.from_numpy(y).float().to(device)  # 转换为 PyTorch 张量
#         x = torch.from_numpy(x).float().to(device)  # 转换为 PyTorch 张量
#         h = torch.exp(-(x * x + y * y) / (2. * sd * sd))
#         h[h < torch.finfo(h.dtype).eps * h.max()] = 0
#         sumh = h.sum()
#         if sumh != 0:
#             win = h / sumh

#         if scale > 1:
#             refA = torch.conv2d(refA.unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze()
#             refB = torch.conv2d(refB.unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze()
#             dist = torch.conv2d(dist.unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze()
#             refA = refA[::2, ::2]
#             refB = refB[::2, ::2]
#             dist = dist[::2, ::2]

#         mu1A = torch.conv2d(refA.unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze()
#         mu1B = torch.conv2d(refB.unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze()
#         mu2 = torch.conv2d(dist.unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze()
#         mu1_sq_A = mu1A * mu1A
#         mu1_sq_B = mu1B * mu1B
#         mu2_sq = mu2 * mu2
#         mu1A_mu2 = mu1A * mu2
#         mu1B_mu2 = mu1B * mu2
#         sigma1A_sq = torch.conv2d((refA * refA).unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze() - mu1_sq_A
#         sigma1B_sq = torch.conv2d((refB * refB).unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze() - mu1_sq_B
#         sigma2_sq = torch.conv2d((dist * dist).unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze() - mu2_sq
#         sigma12_A = torch.conv2d((refA * dist).unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze() - mu1A_mu2
#         sigma12_B = torch.conv2d((refB * dist).unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), padding='valid').squeeze() - mu1B_mu2

#         sigma1A_sq[sigma1A_sq < 0] = 0
#         sigma1B_sq[sigma1B_sq < 0] = 0
#         sigma2_sq[sigma2_sq < 0] = 0

#         gA = sigma12_A / (sigma1A_sq + eps)
#         gB = sigma12_B / (sigma1B_sq + eps)
#         sv_sq_A = sigma2_sq - gA * sigma12_A
#         sv_sq_B = sigma2_sq - gB * sigma12_B

#         gA[sigma1A_sq < eps] = 0
#         gB[sigma1B_sq < eps] = 0
#         sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
#         sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
#         sigma1A_sq[sigma1A_sq < eps] = 0
#         sigma1B_sq[sigma1B_sq < eps] = 0

#         gA[sigma2_sq < eps] = 0
#         gB[sigma2_sq < eps] = 0
#         sv_sq_A[sigma2_sq < eps] = 0
#         sv_sq_B[sigma2_sq < eps] = 0

# if __name__ == '__main__':
def Runevaluator(target,path_save,epoch,data_type,path_ir,path_vi):
    file_list = []
    for fname in os.listdir(path_ir):
        fpath = os.path.join(path_ir, fname)
        if os.path.isfile(fpath):
            file_list.append(fname)
    dataset_name_fi = os.path.join(data_type+ '_'+ target, str(epoch))
    EN = 0
    SD = 0
    SF = 0
    AG = 0
    MI = 0
    MSE = 0
    CC = 0
    PSNR = 0
    SCD = 0
    VIFF = 0
    Qbaf = 0
    SSIM = 0
    new_viff = 0
    count = 0

    for file in file_list:
        irpath = os.path.join(path_ir, file)
        vipath = os.path.join(path_vi, file)
        fipath = os.path.join("./test_result", dataset_name_fi, file)

        vi = torch.from_numpy(image_read_cv2(vipath, 'GRAY')).to(device)
        fi = torch.from_numpy(image_read_cv2(fipath, 'GRAY')).to(device)
        ir = torch.from_numpy(image_read_cv2(irpath, 'GRAY')).to(device)
        EN = EN + Evaluator.EN(fi)
        SD = SD + Evaluator.SD(fi)
        SF = SF + Evaluator.SF(fi)
        AG = AG + Evaluator.AG(fi)
        MI = MI + Evaluator.MI(fi, ir, vi)
        # MSE = MSE + Evaluator.MSE(fi, ir, vi)
        # CC = CC + Evaluator.CC(fi, ir, vi)
        # PSNR = PSNR + Evaluator.PSNR(fi, ir, vi)

        SCD = SCD + Evaluator.SCD(fi, ir, vi)
        VIFF = VIFF + Evaluator.VIFF(fi, ir, vi)
        Qbaf = Qbaf + Evaluator.Qabf(fi, ir, vi)
        SSIM = SSIM + Evaluator.SSIM(fi, ir, vi)
        # new_viff = new_viff + VIFF1(fi, ir, vi)
        count = count + 1
        # print("已计算图片数:",count)
    return EN,SD,SF,AG,SCD,VIFF,MI,SSIM,Qbaf,count