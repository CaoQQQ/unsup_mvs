# -*- coding: utf-8 -*-
# @Time    : 2020/05/29 20:19
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : unsup_loss
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.modules import *
from losses.homography import *


class UnSupLoss(nn.Module):
    def __init__(self):
        super(UnSupLoss, self).__init__()
        self.ssim = SSIM()

    def forward(self, imgs, cams, depth):
        # print('imgs: {}'.format(imgs.shape))
        # print('cams: {}'.format(cams.shape))
        # print('depth: {}'.format(depth.shape))
        '''
        当张量被解绑时，它的指定维度会被拆分成多个张量，这些张量将以一个 Python 列表的形式返回。
        拆分的张量将具有与原始张量相同的大小，除了指定的维度被移除。
例如，   如果有一个形状为 [3, 4, 5] 的张量 x，并且调用 torch.unbind(x, dim=0)，那么将返回一个包含 3 个形状为 [4, 5] 的张量的列表。
        '''
        # imgs 和 cams 使用 torch.unbind 函数进行解绑，得到一个图片列表和一个相机参数列表。
        imgs = torch.unbind(imgs, 1)  # img:[B ref+src 3 H W]
        cams = torch.unbind(cams, 1)  # cams:[B ref+src ... 类似
        # 检查图片和相机参数的数量是否一致。
        assert len(imgs) == len(cams), "Different number of images and projection matrices"  #len返回列表的长度
        # 获取图像的高度和宽度，并记录图像数量。
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_views = len(imgs)
        # 选择参考图像和相机参数
        ref_img = imgs[0]
        # ref_img = F.interpolate(ref_img, scale_factor=0.25, mode='bilinear')
        # ref_img = F.interpolate(ref_img, size=[depth.shape[1], depth.shape[2]])
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        # print('ref_cam: {}'.format(ref_cam.shape))

        # 初始化一些损失变量。
        # reconstr_loss 是重建损失（reconstruction loss）。
        # ssim_loss 是结构相似性损失（structural similarity loss）。
        # smooth_loss 是平滑损失（smoothness loss）。
        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0

        # 对每个视角进行迭代计算。
        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):  # 从第二个视角开始（索引为 1）。
            view_img = imgs[view]
            view_cam = cams[view]
            # print('view_cam: {}'.format(view_cam.shape))
            # view_img = F.interpolate(view_img, scale_factor=0.25, mode='bilinear')
            # view_img = F.interpolate(view_img, size=[depth.shape[1], depth.shape[2]])
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img
            # 使用参考图片的深度图（depth）将当前视角的图片进行逆向变换（inverse warping）
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            warped_img_list.append(warped_img)
            mask_list.append(mask)
            # 计算重建损失（reconstr_loss）和有效掩模（valid_mask）。
            reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=True)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

            # SSIM loss##
            # 对前三个视角计算结构相似性损失（ssim_loss）。
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))
        #print(reconstr_loss)
        ##smooth loss##
        # 计算平滑损失（smooth_loss）。
        self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, 1.0)

        # top-k operates along the last dimension, so swap the axes accordingly
        # 构建重投影损失体积（reprojection volume）并选择前三个最小值。
        reprojection_volume = torch.stack(reprojection_losses).permute(1,2,3,4,0)  # 将重建损失存储为一个损失体积（reprojection volume）。
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)  # 使用 torch.topk 函数选择体积中最小的前三个值和对应的索引。
        top_vals = torch.neg(top_vals)
        # 创建一个与 top_vals 相同大小的掩模（top_mask），将小于 1e4 的值设为 1，其他值设为 0。
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_mask.float()
        # 将 top_vals 乘以 top_mask，将大于 1e4 的值设为 0。
        top_vals = torch.mul(top_vals, top_mask)
        # 计算 top_vals 沿最后一个维度的和，并取平均值，得到 reconstr_loss
        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        #self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + 0.18 * self.smooth_loss
        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + 0.05 * self.smooth_loss
        # 按照un_mvsnet和M3VSNet的设置
        #print(reconstr_loss)
        #self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15

        return self.unsup_loss


def non_zero_mean_absolute_diff(y_true, y_pred, interval):
    """ non zero mean absolute loss for one batch """
    batch_size = y_pred.shape[0]
    interval = interval.reshape(batch_size)
    mask_true = torch.ne(y_true, 0.0).float()
    denom = torch.sum(mask_true, dim=[1,2]) + 1e-7
    masked_abs_error = torch.abs(mask_true * (y_true - y_pred))
    masked_mae = torch.sum(masked_abs_error, dim=[1,2])
    masked_mae = torch.sum((masked_mae.float() / interval.float()) / denom.float())
    return masked_mae


def less_one_percentage(y_true, y_pred, interval):
    """ less one accuracy for one batch """
    batch_size = y_pred.shape[0]
    height = y_pred.shape[1]
    width = y_pred.shape[2]
    interval = interval.reshape(batch_size)
    mask_true = torch.ne(y_true, 0.0).float()
    denom = torch.sum(mask_true) + 1e-7
    interval_image = interval.reshape(batch_size, 1, 1).repeat(1, height, width)
    # print('y_true: {}'.format(y_true.shape))
    # print('y_pred: {}'.format(y_pred.shape))
    abs_diff_image = torch.abs(y_true.float() - y_pred.float()) / interval_image.float()
    less_three_image = mask_true * torch.le(abs_diff_image, 1.0).float()
    return torch.sum(less_three_image) / denom


def less_three_percentage(y_true, y_pred, interval):
    """ less three accuracy for one batch """
    batch_size = y_pred.shape[0]
    height = y_pred.shape[1]
    width = y_pred.shape[2]
    interval = interval.reshape(batch_size)
    mask_true = torch.ne(y_true, 0.0).float()
    denom = torch.sum(mask_true) + 1e-7
    interval_image = interval.reshape(batch_size, 1, 1).repeat(1, height, width)
    abs_diff_image = torch.abs(y_true.float() - y_pred.float()) / interval_image.float()
    less_three_image = mask_true * torch.le(abs_diff_image, 3.0).float()
    return torch.sum(less_three_image) / denom






