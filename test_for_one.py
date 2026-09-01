# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm
from SRutils.Evaluator_deepseeker import Runevaluator
import logging
logging.basicConfig(level=logging.CRITICAL)
import numpy as np

from utils import image_read_cv2,img_save
from nets.Ufuserplus import Ufuserplus




# ../../IF-FILM-main/VLFDataset/Image/IVF/RoadScene/IR
# path_ir=r"../IF-FILM-main/VLFDataset/Image/IVF/M3FD/IR"
# path_vi=r"../IF-FILM-main/VLFDataset/Image/IVF/M3FD/VI"

# path_ir=r"../IF-FILM-main/VLFDataset/Image/MIF/Harvard/MRI"
# path_vi=r"../IF-FILM-main/VLFDataset/Image/MIF/Harvard/T"


# path_ir=r"../LLVIP/infrared/test"
# path_vi=r"../LLVIP/visible/test"

# path_ir=r"../FMB/ir"
# path_vi=r"../FMB/vi"

# path_ir=r"../TNO/ir"
# path_vi=r"../TNO/vi"

# path_ir=r"../M3FD/IR"
# path_vi=r"../M3FD/VI"

# path_ir=r"../Harvard/MRI"
# path_vi=r"../Harvard/T"

# path_ir=r"./dataprocessing/MSRS_train/test/ir"
# path_vi=r"./dataprocessing/MSRS_train/test/vi"
# path_ir=r"../test_img/test_MIF/MRI_SPECT/SPECT"
# path_vi=r"../test_img/test_MIF/MRI_SPECT/MRI"
# path_ir=r"./dataprocessing/detection/ir"
# path_vi=r"./dataprocessing/detection/vi"
path_ir=r"../MSRS-main/detection/ir"
path_vi=r"../MSRS-main/detection/vi"
# path_save=r"test_result/RoadScene4"
# path_save=r"test_result/MSRS_new"
# path_model=r"./test_result/EMMA_03-08-10-53.pth"
# path_model=r"./EMMA_12-12-10-43.pth"
# path_model=r"./EMMA_12-20-11-01.pth"
# path_model=r"newmodels/mirai_2/Smodel/EMMA_07-29-16-37__80.pth"



# path_ir=r"../RoadScene/IR"
# path_vi=r"../RoadScene/VI"

path_model=r"newmodels/mirai_472/Finalmodel/EMMA_03-29-00-10__40.pth"

# path_model=r"newmodels/yuan/EMMA_03-09-04-23.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model=Ufuser().to(device)
# model.load_state_dict(torch.load(path_model))
# model.eval()
model=Ufuserplus().to(device)
model.load_state_dict(torch.load(path_model))
model.eval()
# 需要注意一点我们的输入是vi和ir，而原始模型是ir和vi，是反着的
# 现在是改成vi和ir了，以后要测原始模型还要记得改回去
target = "mirai_472"
data_type = 'MSRS'
path = os.path.join(r"newmodels/",target)
epoch = '667'
# path_save=r"test_result/MSRS_new"
path_save=os.path.join(r"test_result" , data_type + "_" + target,epoch)
# files = get_file_names(path)
i = 1
print("               EN      SD        SF      AG     SCD     VIFF")



    


with torch.no_grad():
    for imgname in tqdm(os.listdir(path_ir)):
    
        IR = image_read_cv2(os.path.join(path_ir, imgname), 'GRAY')[np.newaxis,np.newaxis,...]/255
        VI = image_read_cv2(os.path.join(path_vi, imgname), 'GRAY')[np.newaxis,np.newaxis,...]/255

        h, w = IR.shape[2:]
        h1 = h - h % 32
        w1 = w - w % 32
        h2 = h % 32
        w2 = w % 32

        if h1==h and w1==w: # Image size can be divided by 32
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
print("第{}轮的数据:".format(10*i),'%.2f'%(EN/count),"  ",'%.2f'%(SD/count),"  ",'%.2f'%(SF/count),"  ",'%.2f'%(AG/count),"  ",'%.2f'%(SCD/count),"  ",'%.2f'%(VIFF/count),'%.2f'%(MI/count),"  ",'%.2f'%(Qbaf/count),"  ",'%.2f'%(SSIM/count),"  ",)
