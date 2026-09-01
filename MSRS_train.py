import sys
import os
sys.path.append(os.getcwd())
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist
    
def rgb2y(img):
    # 标准的灰度值转换公式
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

def Im2Patch(img, win, stride=1):
    k = 0
    # 1.480.640
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    # 由于图片是一通道，前面:表示全取，长每128组取一次，宽每128取一次，取出来作为一组。
    # 步长两次，最多两次
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    # print("Shape",patch.shape[1],patch.shape[2])
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            # (1, 16384, 6)
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])
# def Im2Patch(image, patch_size, stride):
#     """
#     将给定的图像分割成重叠的图像块。

#     Args:
#         image (numpy.ndarray): 输入图像，形状为 (通道数, 高度, 宽度)。
#         patch_size (int): 每个图像块的尺寸 (正方形)。
#         stride (int): 滑动窗口的步长。

#     Returns:
#         numpy.ndarray: 包含所有提取的图像块的数组，
#                        形状为 (图像块数量, 通道数, 图像块高度, 图像块宽度)。
#     """
#     channels, height, width = image.shape
#     patches = []

#     for y in range(0, height - patch_size + 1, stride):
#         for x in range(0, width - patch_size + 1, stride):
#             patch = image[:, y:y + patch_size, x:x + patch_size]
#             patches.append(patch)

#     return np.array(patches).transpose(1,2,3,0)

def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

data_name="MSRS_train"
img_size=128   #patch size
stride=200 #patch stride
# IR_files = sorted(get_img_file(r"../M3FD/Ir"))
# VIS_files   = sorted(get_img_file(r"../M3FD/Vis"))

IR_files = sorted(get_img_file(r"./dataprocessing/MSRS_train/train/ir"))
VIS_files   = sorted(get_img_file(r"./dataprocessing/MSRS_train/train/vi"))


# Mask_files = sorted(get_img_file(r"./dataprocessing/M3FD/train/Mask"))
assert len(IR_files) == len(VIS_files)
h5_path=os.path.join('./Data', data_name+"_"+str(img_size)+"_"+str(stride)+"-1-0FB"+'.h5')
h5f = h5py.File(h5_path,'w')
h5_ir = h5f.create_group('ir_patchs')
h5_vis = h5f.create_group('vis_patchs')
train_num=0
for i in tqdm(range(len(IR_files))):
        I_VIS = imread(VIS_files[i]).astype(np.float32).transpose(2,0,1)/255. # [3, H, W] Uint8->float32
        I_VIS = rgb2y(I_VIS) # [1, H, W] Float32
        # I_IR = imread(IR_files[i]).astype(np.float32)[None, :, :]/255.
        I_IR = imread(IR_files[i]).astype(np.float32)[None, :, :]/255.   # [1, H, W] Float32
        # print(I_IR.shape)
        # I_Mask = imread(Mask_files[i]).astype(np.float32)[None, :, :]/255
        # 这一段是为了迎合扩散模型0-1的分布
        # crop    
        I_IR_Patch_Group = Im2Patch(I_IR,img_size,stride)
        I_VIS_Patch_Group = Im2Patch(I_VIS, img_size, stride)  # (3, 256, 256, 12)
        # I_Mask_Patch_Group = Im2Patch(I_Mask,img_size, stride)
        # print(I_IR_Patch_Group.shape) 
        # (3, 768, 1024)
        # (1, 768, 1024)
        # (4, 1, 256, 256)
        # (1, 128, 128, 6)
        # (1, 256, 256, 4)
        
        for ii in range(I_IR_Patch_Group.shape[-1]):
            bad_IR = is_low_contrast(I_IR_Patch_Group[0,:,:,ii])
            bad_VIS = is_low_contrast(I_VIS_Patch_Group[0,:,:,ii])
            # bad_Mask = is_low_contrast(I_Mask_Patch_Group[0,:,:,ii])
            # Determine if the contrast is low
            if not (bad_IR or bad_VIS):
                avl_IR= I_IR_Patch_Group[0,:,:,ii]  #  available IR
                avl_VIS= I_VIS_Patch_Group[0,:,:,ii]
                # avl_Mask= I_Mask_Patch_Group[0,:,:,ii]
                avl_IR=avl_IR[None,...]
                avl_VIS=avl_VIS[None,...]
                # avl_Mask=avl_Mask[None,...]
                avl_IR = avl_IR*2-1
                avl_VIS = avl_VIS*2-1
                h5_ir.create_dataset(str(train_num),     data=avl_IR, 
	                            dtype=avl_IR.dtype,   shape=avl_IR.shape)
                h5_vis.create_dataset(str(train_num),    data=avl_VIS, 
	                            dtype=avl_VIS.dtype,  shape=avl_VIS.shape)
                # h5_mask.create_dataset(str(train_num),    data=avl_Mask, 
	            #                 dtype=avl_Mask.dtype,  shape=avl_Mask.shape)                

                train_num += 1        

h5f.close()

with h5py.File(os.path.join('./Data', data_name+"_"+str(img_size)+"_"+str(stride)+"-1-0FB"+'.h5'),"r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name) 