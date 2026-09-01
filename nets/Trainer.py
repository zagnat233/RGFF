import os
import numpy as np
from typing import Dict, Any, Callable,Optional
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from utils import Result_Save,UNGANLoss,loss_fusion_f,loss_fusion_m,loss_fusion,r1_penalty
from pytorch_msssim import SSIM
ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)
class F2Trainer(nn.Module):
    def __init__(
        self,
        fusion_model,
        fusion_model_TP,
        generator,
        d_vi,
        d_ir,
        result_save_fn=Result_Save,      
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.Fu_TP = fusion_model_TP
        self.Fu = fusion_model
        self.G = generator
        self.D = nn.ModuleDict({"v": d_vi, "i": d_ir})
        self.gan_mse = nn.MSELoss()
        self.CE_loss= UNGANLoss(gan_type ='vanilla').to(device)
        self.mse = nn.MSELoss()
        self.loss_m = loss_fusion_m().to(device)
        self.loss_f = loss_fusion_f().to(device)
        self.loss_g = loss_fusion().to(device)
        self.sobelxy = Sobelxy()
        self.Result_Save = result_save_fn
        self.device = device
        self.save_root = "newmodels"
        self.gamma = 10
    @staticmethod
    def _set_requires_grad(net: nn.Module, flag: bool):
        for p in net.parameters():
            p.requires_grad_(flag)
    @staticmethod
    def _as_scalar_loss(x):
        return x[0] if isinstance(x, (tuple, list)) else x
    @staticmethod
    def _rand_uniform_like(x, low, high):
        return torch.empty_like(x).uniform_(low, high)
    @torch.no_grad()
    def fuse_Fm(self, data_VIS, data_IR):

        self.Fu.eval()
        return self.Fu(data_VIS, data_IR)
    def _soft_scalar_map_like(self, x: torch.Tensor, low: float, high: float):
        """
        每个 batch sample 采一个随机标量，然后扩展成和 x 同形状的 map
        x: [B, C, H, W]
        """
        shape = [x.size(0)] + [1] * (x.dim() - 1)   # [B,1,1,1]
        t = torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(low, high)
        return t.expand_as(x)


    def _feature_matching_loss(self, fake_feat, real_feat):
        """
        支持判别器返回单层特征或多层特征列表
        """
        if isinstance(fake_feat, (list, tuple)):
            loss = 0.0
            for f, r in zip(fake_feat, real_feat):
                loss = loss + F.l1_loss(f, r.detach())
            return loss / len(fake_feat)
        else:
            return F.l1_loss(fake_feat, real_feat.detach())


    def _down_to_score_size(self, ref_img: torch.Tensor, score_map: torch.Tensor):
        """
        把参考图像降采样到和 PatchGAN score map 相同的空间尺寸
        ref_img:   [B,1,H,W]
        score_map: [B,1,h,w]
        """
        return F.interpolate(
            ref_img,
            size=score_map.shape[-2:],
            mode='bilinear',
            align_corners=False
        )


    def _edge_aware_response_smooth_loss(
        self,
        score_map: torch.Tensor,
        ref_img: torch.Tensor,
        alpha: float = 10.0,
    ):
        """
        对判别器响应图做边缘感知平滑：
        - 在 ref_img 的平坦区域，鼓励 score_map 更平滑
        - 在 ref_img 的边缘区域，放松约束

        score_map: [B,1,h,w]
        ref_img:   [B,1,H,W]
        """
        ref_img_ds = self._down_to_score_size(ref_img, score_map)

        score_grad = torch.abs(self.sobelxy(score_map))
        ref_grad = torch.abs(self.sobelxy(ref_img_ds))

        weight = torch.exp(-alpha * ref_grad)
        return (weight * score_grad).mean()
    def set_phase_model_teacher(self):
        self.Fu.eval()
        self._set_requires_grad(self.Fu, False)

        self.G.eval()
        self._set_requires_grad(self.G, False)

        self.Fu_TP.train()
        self._set_requires_grad(self.Fu_TP, True)
    def save_triplet_images(
        self,
        *,
        epoch: int,
        batch_idx: int,
        indices,
        name: str,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        save_every: int = 1,
        save_first_batches: int = 2,
        max_save: int = 8,
        subdir: str = "Decouplepics",
    ):
        # 触发条件：每 save_every 个 epoch，且只存前 save_first_batches 个 batch
        if ((epoch + 1) % save_every != 0) or (batch_idx >= save_first_batches):
            return

        save_dir = os.path.join(self.save_root, name, subdir, str(epoch + 1))
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            Bsz = A.shape[0]
            n = min(Bsz, max_save)

            for j in range(n):
                a = A[j].detach().squeeze().float().clamp(0, 1).cpu().numpy()
                b = B[j].detach().squeeze().float().clamp(0, 1).cpu().numpy()
                c = C[j].detach().squeeze().float().clamp(0, 1).cpu().numpy()

                temp = np.concatenate([a, b, c], axis=1)          # (H, 3W)
                temp = (temp * 255.0).round().astype(np.uint8)

                fname = f"{int(indices[j])}.png" if indices is not None else f"{j}.png"
                cv2.imwrite(os.path.join(save_dir, fname), temp)

    def save_model_and_eval(
        self,
        *,
        epoch: int,
        num_epochs: int,
        name: str,
        timestamp: str,
        save_every: int = 10,
        module: nn.Module = None,          
        only_last: bool = False,           
        subdir: str = "Smodel",
        final_subdir: str = "Finalmodel",
        prefix: str = "EMMA",
        do_eval: bool = True,             
        text_filename: str = "Smodelresult.txt",
        write_header_at: int = 10,
        data_type,
    ):
        if module is None:
            module = self.Fu

        # 触发条件
        if only_last:
            if (epoch + 1) != num_epochs:
                return
        else:
            if (epoch + 1) % save_every != 0:
                return

        sort = epoch + 1

        # 路径
        sub_dir = os.path.join(self.save_root, name, subdir)
        fin_dir = os.path.join(self.save_root, name, final_subdir)
        os.makedirs(sub_dir, exist_ok=True)
        os.makedirs(fin_dir, exist_ok=True)

        ckpt_name = f"{prefix}_{timestamp}__{sort}.pth"
        if sort == num_epochs:
            modelpath = os.path.join(fin_dir, ckpt_name)
        else:
            modelpath = os.path.join(sub_dir, ckpt_name)

        torch.save(module.state_dict(), modelpath)

        if do_eval and (self.Result_Save is not None):
            text_path = os.path.join(self.save_root, name, text_filename)
            try:
                with open(text_path, "a", encoding="utf-8") as f:
                    if sort == write_header_at:
                        f.write("             EN      SD      SF     AG     SCD    VIFF   MI   Qbaf   SSIM  \n")
            except IOError as e:
                print(f"创建/写入结果文件时发生错误：{e}")
            if data_type == 'all':
                data_type_t = 'MSRS'
                self.Result_Save(sort, modelpath, text_path, name, data_type_t)
                data_type_t = 'FMB'
                self.Result_Save(sort, modelpath, text_path, name, data_type_t)
                data_type_t = 'M3FD'
                self.Result_Save(sort, modelpath, text_path, name, data_type_t)
                data_type_t = 'Harvard'
                self.Result_Save(sort, modelpath, text_path, name, data_type_t)
            else:
                self.Result_Save(sort, modelpath, text_path, name, data_type)
    def train_step_fusion_only(
        self,
        data_VIS: torch.Tensor,
        data_IR: torch.Tensor,
        *,
        opt_fu,               
        # 可选：封装保存拼接图
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
        indices=None,
        name: str = "",
        save_images: bool = False,
        save_every: int = 1,
        save_first_batches: int = 2,
        subdir: str = "Hasjipics",
        max_save: int = 8,
    ):
        # 该阶段只训练 Fu
        self.Fu.train()
        self._set_requires_grad(self.Fu, True)

        F_m = self.Fu(data_VIS, data_IR)

        loss_t = self._as_scalar_loss(self.loss_m(data_VIS, data_IR, F_m))
        opt_fu.zero_grad(set_to_none=True)
        loss_t.backward()
        opt_fu.step()

        # 可选保存拼接图（统一走通用函数）
        if save_images and (epoch is not None) and (batch_idx is not None) and name:
            self.save_triplet_images(
                epoch=epoch,
                batch_idx=batch_idx,
                indices=indices,
                name=name,
                A=data_IR,      # 左：IR
                B=data_VIS,     # 中：VIS
                C=F_m,          # 右：融合结果
                save_every=save_every,
                save_first_batches=save_first_batches,
                max_save=max_save,
                subdir=subdir,
            )

        return {"loss_total": loss_t.detach()}
    def structural_loss(self,fused, visible):
        loss= 1 - ssim_loss(fused, visible)
        return loss
    def compute_De_loss(self, data_VIS, data_IR, F_m: torch.Tensor,FRM_state):
        if FRM_state == True:
            VI_P, IR_P = self.G(data_VIS, data_IR,F_m) 
        else:
            VI_P, IR_P = self.G(F_m) 
        de_loss_vi = self.gan_mse(data_VIS, VI_P)
        de_loss_ir = self.gan_mse(data_IR, IR_P)
        de_loss_total = de_loss_vi + de_loss_ir
        logs = {
            "de_loss_total": de_loss_total.detach(),
        }
        return de_loss_total,logs,VI_P.detach(),IR_P.detach()
    def compute_d_loss(self, data_VIS, data_IR, F_m: torch.Tensor, FRM_state,
                lambda_adv_vi = 0.02,
                lambda_adv_ir = 0.02,
                lambda_rsp_vi = 0.01,
                lambda_rsp_ir = 0.01,
                alpha_rsp = 10.0):
        with torch.no_grad():
            if FRM_state == True:
                VI_P, IR_P = self.G(data_VIS, data_IR, F_m) 
            else:
                VI_P, IR_P = self.G(F_m) 
        # 这里判别器开始
        real_logit_vi = self.D["v"](data_VIS)   # [B,1,h,w]
        fake_logit_vi = self.D["v"](VI_P)       # [B,1,h,w]

        # soft labels：每张图一个随机标量
        real_t_vi = self._soft_scalar_map_like(real_logit_vi, 0.85, 1.0)
        fake_t_vi = self._soft_scalar_map_like(fake_logit_vi, 0.0, 0.15)

        # 对抗强度约束
        d_adv_vi = (
            F.l1_loss(real_logit_vi, real_t_vi) +
            F.l1_loss(fake_logit_vi, fake_t_vi)
        )

        # 判别器响应图平滑正则
        d_rsp_vi = (
            self._edge_aware_response_smooth_loss(real_logit_vi, data_VIS, alpha=alpha_rsp) +
            self._edge_aware_response_smooth_loss(fake_logit_vi, VI_P, alpha=alpha_rsp)
        )

        d_loss_vi = lambda_adv_vi * d_adv_vi + lambda_rsp_vi * d_rsp_vi

        # ================= IR branch =================
        real_logit_ir = self.D["i"](data_IR)
        fake_logit_ir = self.D["i"](IR_P)

        real_t_ir = self._soft_scalar_map_like(real_logit_ir, 0.85, 1.0)
        fake_t_ir = self._soft_scalar_map_like(fake_logit_ir, 0.0, 0.15)

        d_adv_ir = (
            F.l1_loss(real_logit_ir, real_t_ir) +
            F.l1_loss(fake_logit_ir, fake_t_ir)
        )

        d_rsp_ir = (
            self._edge_aware_response_smooth_loss(real_logit_ir, data_IR, alpha=alpha_rsp) +
            self._edge_aware_response_smooth_loss(fake_logit_ir, IR_P, alpha=alpha_rsp)
        )

        d_loss_ir = lambda_adv_ir * d_adv_ir + lambda_rsp_ir * d_rsp_ir

        d_loss_total = d_loss_vi + d_loss_ir
        return d_loss_total

    def compute_g_loss(self, data_VIS, data_IR, F_m: torch.Tensor,FRM_state,
                lambda_adv_vi = 0.02,
                lambda_adv_ir = 0.02,
                lambda_rsp_vi = 0.01,
                lambda_rsp_ir = 0.01,
                lambda_cycle = 0.1,
                alpha_rsp = 10.0):

        # F_m 是 no_grad 得到的常量，这里不用再 detach
        if FRM_state == True:
            VI_Pt, IR_Pt = self.G(data_VIS, data_IR, F_m) 
        else:
            VI_Pt, IR_Pt = self.G(F_m) 

        F_cycle = self.Fu(VI_Pt, IR_Pt)
        loss_cycle = self.loss_g(F_cycle, F_m)  
        fake_logit_vi, fake_feat_vi = self.D["v"](VI_Pt, return_features=True)
        real_logit_vi, real_feat_vi = self.D["v"](data_VIS, return_features=True)
        fake_logit_ir, fake_feat_ir = self.D["i"](IR_Pt, return_features=True)
        real_logit_ir, real_feat_ir = self.D["i"](data_IR, return_features=True)

        g_t_vi = self._soft_scalar_map_like(fake_logit_vi, 0.85, 1.0)
        g_t_ir = self._soft_scalar_map_like(fake_logit_ir, 0.85, 1.0)

        g_adv_vi = F.l1_loss(fake_logit_vi, g_t_vi)
        g_adv_ir = F.l1_loss(fake_logit_ir, g_t_ir)

        # ---------- response smoothness on fake ----------
        g_rsp_vi = self._edge_aware_response_smooth_loss(fake_logit_vi, VI_Pt, alpha=alpha_rsp)
        g_rsp_ir = self._edge_aware_response_smooth_loss(fake_logit_ir, IR_Pt, alpha=alpha_rsp)

        # F.l1_loss(self.sobelxy(fake_logit_vi), self.sobelxy(g_t_vi))
        # F.l1_loss(self.sobelxy(fake_logit_ir), self.sobelxy(g_t_ir)) 

        # gan_loss_vi = self.loss_g(fake_logit_vi, g_t_vi)
        # gan_loss_ir = self.loss_g(fake_logit_ir, g_t_ir)

# + F.l1_loss(fake_logit_vi,g_t_vi)
# + F.l1_loss(fake_logit_ir,g_t_ir)

        loss_vi = self.gan_mse(data_VIS, VI_Pt)
        loss_ir = self.gan_mse(data_IR, IR_Pt)
        loss_total = loss_vi + loss_ir

        # fm_loss_vi = self.gan_mse(fake_feat_vi, real_feat_vi.detach())
        # fm_loss_ir = self.gan_mse(fake_feat_ir, real_feat_ir.detach())

        g_loss_total = (
            + lambda_cycle * loss_cycle
            + loss_total
            + lambda_adv_vi * g_adv_vi
            + lambda_adv_ir * g_adv_ir
            + lambda_rsp_vi * g_rsp_vi
            + lambda_rsp_ir * g_rsp_ir
        )
            # + lambda_fm_vi * fm_loss_vi
            # + lambda_fm_ir * fm_loss_ir
            # + loss_total
        logs = {
            "g_loss_total": g_loss_total.detach(),
            "loss_gan": (lambda_adv_vi * g_adv_vi + lambda_adv_ir * g_adv_ir).detach(),
        }
        return g_loss_total, logs, VI_Pt.detach(), IR_Pt.detach()

    def train_step(
        self,
        data_VIS: torch.Tensor,
        data_IR: torch.Tensor,
        *,
        opt_d_vi,
        opt_d_ir,
        opt_g,
        adv_state = True,
        FRM_state = True,
    ):
        if adv_state == True:
            F_m = self.fuse_Fm(data_VIS, data_IR)
            # F_m = torch.max(data_VIS, data_IR)
            # ----- D step (1x) -----
            # for _ in range(2):
            self.G.eval();
            self.D["v"].train(); self.D["i"].train()
            self._set_requires_grad(self.G, False)
            self._set_requires_grad(self.D["v"], True)
            self._set_requires_grad(self.D["i"], True)


            opt_d_vi.zero_grad(set_to_none=True)
            opt_d_ir.zero_grad(set_to_none=True)
            d_loss = self.compute_d_loss(data_VIS, data_IR, F_m=F_m,FRM_state = FRM_state)
            d_loss.backward()
            opt_d_vi.step(); opt_d_ir.step()

            # ----- G step (1x) -----
            self.G.train();
            self.D["v"].eval();  self.D["i"].eval()
            self._set_requires_grad(self.D["v"], False)
            self._set_requires_grad(self.D["i"], False)
            self._set_requires_grad(self.G, True)

            opt_g.zero_grad(set_to_none=True)
            g_loss, logs, VI_Pt, IR_Pt = self.compute_g_loss(data_VIS, data_IR, F_m=F_m,FRM_state=FRM_state)
            g_loss.backward()
            opt_g.step()

            logs["d_loss"] = d_loss.detach()
            log_float = {k: float(v.item()) for k, v in logs.items()
                        if torch.is_tensor(v) and v.numel() == 1}
        else:
            F_m = self.fuse_Fm(data_VIS, data_IR)
            # F_m = torch.max(data_VIS, data_IR)
            self.G.train()
            self._set_requires_grad(self.G, True)
            opt_g.zero_grad(set_to_none=True)
            de_loss_total,logs,VI_Pt,IR_Pt = self.compute_De_loss(data_VIS, data_IR,F_m=F_m,FRM_state =FRM_state)
            log_float = {k: float(v.item()) for k, v in logs.items()
                        if torch.is_tensor(v) and v.numel() == 1}
            de_loss_total.backward()
            opt_g.step();            
        return log_float, F_m, VI_Pt, IR_Pt
    # ------------------ 核心：训练 Fu（model）一步 ------------------
    def train_step_model_teacher(
        self,
        data_VIS: torch.Tensor,
        data_IR: torch.Tensor,
        *,
        opt_fu_tp,
        lambda_f=1.0,
        lambda_strcut=0.5,
        FRM_state,
    ) -> Dict[str, float]:
        # forward
        F_m = self.Fu(data_VIS, data_IR)     
        if FRM_state == True:
            V_d, I_d = self.G(data_VIS, data_IR,F_m)
        else:
            V_d, I_d = self.G(F_m)
        
        F_cycle = self.Fu_TP(V_d, I_d)
        # F_cycle = self.Fu_TP(data_VIS, data_IR)
        # losses
        # loss_v = self._as_scalar_loss(self.loss_g(V_d, data_VIS))
        # loss_i = self._as_scalar_loss(self.loss_g(I_d, data_IR))
        # loss_cycle = self._as_scalar_loss(self.loss_g(F_cycle, F_m.detach()))
        loss_F = self._as_scalar_loss(self.loss_f(data_VIS, data_IR,F_cycle))
        # loss_strct = self.structural_loss(data_VIS,F_cycle) + self.structural_loss(data_IR,F_cycle)
        loss_total =  lambda_f*loss_F 
        # + lambda_t * loss_cycle + lambda_f*loss_f
        # + lambda_strcut*loss_strct
        opt_fu_tp.zero_grad(set_to_none=True)
        loss_total.backward()
        opt_fu_tp.step()
        return {
            "loss_total": float(loss_total.detach().item()),
            "loss_f": float(lambda_f*loss_F.detach().item()),
            "F_cycle": F_cycle.detach(),
            "V_d": V_d.detach(),
            "I_d": I_d.detach(),
        }
    
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