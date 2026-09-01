import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append(os.getcwd())
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')  
import logging
logging.basicConfig(level=logging.CRITICAL)
import numpy as np
from nets.Trainer import F2Trainer
from nets.Unet258 import FRMNet
from nets.Unet666 import DecoupleNet
from nets.Ufuserplus import Ufuserplus
from SRutils.NLayerDiscriminator_arch import NLayerDiscriminator
from utils import H5Dataset
import shutil

name = "mirai_477"
data_type = 'all'
# 'M3FD' 'Harvard' 'MSRS' 'FMB'
if not os.path.exists(os.path.join("newmodels/",name)):
            os.makedirs(os.path.join("newmodels/",name))

adv_state = True
FRM_state = True


num_epochs_first = 80
num_epochs_second = 40
num_epochs_third = 40

lr = 1e-4
alpha=0.1
batch_size = 4

backup_dir = os.path.join("newmodels/", name)
current_file = os.path.abspath(__file__)
net = "nets/Trainer.py"   
util = "./utils.py"  
os.makedirs(backup_dir, exist_ok=True)

backup_path = os.path.join(backup_dir, os.path.basename(current_file))
shutil.copy2(current_file, backup_path)
shutil.copy2(net, os.path.join(backup_dir, os.path.basename(net))) 
shutil.copy2(util, os.path.join(backup_dir, os.path.basename(util))) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model=Ufuserplus().to(device)
TP_model=Ufuserplus().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

optimizer_TP = torch.optim.Adam(TP_model.parameters(), lr=lr, weight_decay=0)
scheduler_TP = torch.optim.lr_scheduler.StepLR(optimizer_TP, step_size=20, gamma=0.5)

Discrime_vi = NLayerDiscriminator().to(device)
Discrime_ir = NLayerDiscriminator().to(device)

if FRM_state == True:
    Generate = FRMNet().to(device)
else:
    Generate = DecoupleNet().to(device)

optimizer_g = torch.optim.Adam(Generate.parameters(), lr=1e-5, weight_decay=0)

optimizer_d_vi = torch.optim.Adam(Discrime_vi.parameters(), lr=1e-5, weight_decay=0)
optimizer_d_ir = torch.optim.Adam(Discrime_ir.parameters(), lr=1e-5, weight_decay=0)

scheduler_d_vi = torch.optim.lr_scheduler.StepLR(optimizer_d_vi, step_size=20, gamma=0.5)
scheduler_d_ir = torch.optim.lr_scheduler.StepLR(optimizer_d_ir, step_size=20, gamma=0.5)
scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.5)

trainer = F2Trainer(fusion_model=model,
        fusion_model_TP=TP_model,
        generator=Generate,
        d_vi=Discrime_vi,
        d_ir=Discrime_ir,
        ).to(device)

trainloader = DataLoader(H5Dataset(r"Data/MSRS_train_128_200.h5"),batch_size=batch_size, shuffle=True, num_workers=0)
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

prev_time = time.time()

base = os.path.join("newmodels", name)
for sub in ["Smodel", "Hajimodel", "FDmodel"]:
    os.makedirs(os.path.join(base, sub), exist_ok=True)

# model_path = "newmodels/mirai_40/Hajimodel/EMMA_11-02-11-37__80.pth"
model_path = "newmodels/mirai_238/Hajimodel/EMMA_03-09-15-25__80.pth"
model.load_state_dict(torch.load(model_path))
model.train()

# for epoch in range(num_epochs_first):
#     for i, (data_IR, data_VIS, index) in enumerate(trainloader):
#         data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

#         out = trainer.train_step_fusion_only(
#             data_VIS, data_IR,
#             opt_fu=optimizer,
#             epoch=epoch, batch_idx=i, indices=index, name=name,
#             save_images=True,                 
#             save_every=1,
#             save_first_batches=2,             
#             subdir="Hasjipics",
#             max_save=min(8, data_IR.shape[0]),
#         )

#         batches_done = epoch * len(trainloader) + i
#         batches_left = num_epochs_first * len(trainloader) - batches_done
#         time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
#         prev_time = time.time()

#         print(
#             "[Epoch %d/%d] [Batch %d/%d] [loss_total: %.4f]  ETA: %.10s"
#             % (epoch + 1, num_epochs_first, i, len(trainloader), out["loss_total"].item(), str(time_left))
#         )

#     trainer.save_model_and_eval(
#         epoch=epoch,
#         num_epochs=num_epochs_first,
#         name=name,
#         timestamp=timestamp,
#         save_every=10,
#         module=trainer.Fu,
#         subdir="Hajimodel",
#         final_subdir="Hajimodel",
#         prefix="EMMA",
#         do_eval=True,
#         text_filename="Hajimodelresulit.txt",
#         write_header_at=10,
#         data_type=data_type,
#     )
#     scheduler.step()


for epoch in range(num_epochs_second):
    for i, (data_IR, data_VIS, index) in enumerate(trainloader):
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        batchsize, channels, rows, columns = data_IR.shape
        log,F_m,VI_Pt,IR_Pt = trainer.train_step(data_VIS, data_IR,
            opt_d_vi=optimizer_d_vi,
            opt_d_ir=optimizer_d_ir,
            opt_g=optimizer_g,
            adv_state= adv_state,
            FRM_state = FRM_state,)

        batches_done = epoch * len(trainloader) + i
        batches_left = num_epochs_second * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
        prev_time = time.time()
        if adv_state == True:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [g_loss_total: %.4f]  [loss_gan: %.4f] [d_loss: %.4f]  ETA: %.10s"
                % (
                    epoch+1, num_epochs_second, i, len(trainloader),
                    log["g_loss_total"], 
                    log["loss_gan"], log["d_loss"],
                    str(time_left),
                )
            )
        else:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [de_loss_total: %.4f]  ETA: %.10s"
                % (
                    epoch+1, num_epochs_second, i, len(trainloader),
                    log["de_loss_total"], 
                    str(time_left),
                )
            )
        trainer.save_triplet_images(
            epoch=epoch,
            batch_idx=i,
            indices=index,
            name=name,
            A=F_m,
            B=VI_Pt,
            C=IR_Pt,
            save_every=1,
            save_first_batches=2,
            max_save=min(8, data_IR.shape[0]),
            subdir="Decouplepics",
        )
    trainer.save_model_and_eval(
        epoch=epoch, num_epochs=num_epochs_second, name=name, timestamp=timestamp,
        module=Generate, only_last=True,
        subdir="FDmodel", final_subdir="FDmodel",
        prefix="Generate", do_eval=False,data_type=data_type,
    )
    scheduler_g.step()
    scheduler_d_ir.step()
    scheduler_d_vi.step()


# Ge_model_path = "newmodels/mirai_444/FDmodel/Generate_03-25-18-09__40.pth"
# Ge_model_path = "newmodels/mirai_449/FDmodel/Generate_03-25-16-27__40.pth"
# Ge_model_path = "newmodels/mirai_450/FDmodel/Generate_03-26-05-01__40.pth"
# Generate.load_state_dict(torch.load(Ge_model_path))

# 注意这里会改变冻结开放的逻辑
trainer.set_phase_model_teacher()


for epoch in range(num_epochs_third):
    for i, (data_IR, data_VIS, index) in enumerate(trainloader):
        data_VIS = data_VIS.cuda(non_blocking=True)
        data_IR  = data_IR.cuda(non_blocking=True)
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
        out = trainer.train_step_model_teacher(
            data_VIS, data_IR,
            opt_fu_tp=optimizer_TP,
            lambda_f=1.0,
            FRM_state = FRM_state,
        )
        trainer.save_triplet_images(
            epoch=epoch, batch_idx=i, indices=index, name=name,
            A=out["F_cycle"], B=out["V_d"], C=out["I_d"],    
            save_every=1, save_first_batches=2, max_save=min(8, data_IR.shape[0]),
            subdir="Startpics",
        )
        batches_done = epoch * len(trainloader) + i + 1
        batches_total = num_epochs_third * len(trainloader)
        batches_left = max(0, batches_total - batches_done)

        iter_time = time.time() - prev_time
        prev_time = time.time()
        time_left = datetime.timedelta(seconds=int(batches_left * iter_time))
        print(
            "[Epoch %d/%d] [Batch %d/%d] "
            "[loss_total: %.4f] ETA: %.10s"
            % (
                epoch + 1,
                num_epochs_third,
                i,
                len(trainloader),
                out["loss_total"],
                str(time_left),
            )
        )
    # 注意这里改了保存的模型，是TP
    trainer.save_model_and_eval(
        epoch=epoch, num_epochs=num_epochs_third,
        name=name, timestamp=timestamp,
        save_every=10,
        module=trainer.Fu_TP,
        subdir="Smodel", final_subdir="Finalmodel",
        prefix="EMMA",
        do_eval=True,
        data_type=data_type,
    )
    scheduler_TP.step()




