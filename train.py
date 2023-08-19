# -*- coding: utf-8 -*-
# @Time    : 2020/04/16 16:56
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : train
# @Software: PyCharm
import gc
import os, sys, time, logging, datetime
import errno
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# from dataset.dtu_yao2 import MVSDataset
from dataset.dtu import DTUDataset
from models.network import CVPMVSNet, sL1_loss, MSE_loss
from argParser import getArgsParser, checkArgs
from utils import *
from losses.unsup_loss import *
#from losses.unsup_seg_loss import *
#from models.augmentations import random_image_mask, aug_loss

# Arg parser
parser = getArgsParser()
args = parser.parse_args()
assert args.mode == 'train'
checkArgs(args)
# 设置 (CPU) 生成随机数的种子 设置种子的用意是一旦固定种子，后面依次生成的随机数其实都是固定的。
torch.manual_seed(args.seed)
# 设置 (GPU) 生成随机数的种子
torch.cuda.manual_seed(args.seed)
# 如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true 可以增加运行效率；
torch.backends.cudnn.benchmark = True

# Check checkpoint directory
# 如果不存在ckpt保存的文件夹，那么就自己创建一个
if not os.path.exists(args.logckptdir + args.info.replace(" ", "_")):
    try:
        os.makedirs(args.logckptdir + args.info.replace(" ", "_"))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# logger 一些日志的初始化操作
logger = logging.getLogger()
logger.setLevel(logging.INFO)
curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
log_path = args.loggingdir + args.info.replace(" ", "_") + "/"
if not os.path.isdir(args.loggingdir):
    os.mkdir(args.loggingdir)
if not os.path.isdir(log_path):
    os.mkdir(log_path)
log_name = log_path + curTime + '.log'
logfile = log_name
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fileHandler = logging.FileHandler(logfile, mode='a')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
logger.info("Logger initialized.")
logger.info("Writing logs to file:" + logfile)

settings_str = "All settings:\n"
line_width = 30
for k, v in vars(args).items():
    settings_str += '{0}: {1}\n'.format(k, v)
logger.info(settings_str)
logger.info(args.log)#日志的主要内容

# summary tensorboard有关的算法
summary_writer = SummaryWriter(args.summarydir)

# 分布训练
torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend="nccl", init_method="env://")
dist.barrier()
# world_size=dist.get_world_size()

#  导入数据集
if args.dataset == 'dtu':
    train_dataset = DTUDataset(args, logger=logger)
else:
    raise RuntimeError('args.dataset is not supported.')
train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                    rank=dist.get_rank())  # 切分训练数据集
train_loader = DataLoader(train_dataset, args.batch_size, num_workers=11, pin_memory=True, sampler=train_sampler,
                          drop_last=True)
# train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16, drop_last=True)


# Network
model = CVPMVSNet(args)
logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

# model = nn.DataParallel(model)
# model.cuda()
model.train()

device = torch.device("cuda", args.local_rank)
model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

# Loss
criterion = UnSupLoss().cuda()
#criterion_seg = UnSupSegLoss(args).cuda()
#seg_viz = SegVisualizer(K=args.seg_clusters)
#criterion_aug = aug_loss

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# load network parameters
sw_path = ''
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    logger.info("Resuming or testing...")
    saved_models = [fn for fn in os.listdir(sw_path) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # use the latest checkpoint file
    loadckpt = os.path.join(sw_path, saved_models[-1])
    logger.info("Resuming " + loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    logger.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])

# Start training
logger.info("start at epoch {}".format(start_epoch))
logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))



# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)
    last_loss = None
    this_loss = None

    # global_step = 0

    for epoch_idx in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch_idx)
        logger.info('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(train_loader) * epoch_idx
        #adjust_parameters(epoch_idx)

        # training
        for batch_idx, sample in enumerate(train_loader):
            start_time = time.time()
            # global_step += 1
            global_step = len(train_loader) * epoch_idx + batch_idx
            do_summary = (global_step % args.summary_freq == 0)
            loss, scalar_outputs, image_outputs, standard_loss, depth_est_list = train_sample(sample,do_summary)
            #if global_step % 10 == 0:
                #logger.info(scalar_outputs)
            if do_summary and args.local_rank == 0:
                save_scalars(summary_writer, 'train', scalar_outputs, global_step)
                # print("scalars",scalar_outputs)
                save_images(summary_writer, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            '''
            augment_loss, scalar_outputs, image_outputs, = train_sample_aug(sample, depth_est_list, do_summary)
            if do_summary and args.local_rank == 0:
                save_scalars(summary_writer, 'train', scalar_outputs, global_step)
                save_images(summary_writer, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            '''
            logger.info(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, standard loss = {:.3f},time = {:.3f}, global_step = {:.3f}'.format
                (
                    epoch_idx,
                    args.epochs,
                    batch_idx,
                    len(train_loader),
                    loss,
                    standard_loss,
                    time.time() - start_time,
                    global_step
                )
            )
            lr_scheduler.step()

        if (epoch_idx + 1) % args.save_freq == 0 and args.local_rank == 0:
            # if global_step % args.save_freq == 0 and args.local_rank == 0:
            torch.save({
                'epoch': epoch_idx,
                'iter': global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:08}.ckpt".format(args.logckptdir, epoch_idx))
        gc.collect()


def train_sample(sample, detailed_summary=False):
    optimizer.zero_grad()  # 先将梯度归零（optimizer.zero_grad()）

    sample_cuda = tocuda(sample)
    ref_depths = sample_cuda["ref_depths"]
    cams = sample_cuda["cams"]

    # 输入参数：forward(self, ref_img, src_imgs, ref_in, src_in, ref_ex, src_ex, depth_min, depth_max)
    outputs = model(
        sample_cuda["ref_img"].float(),
        sample_cuda["src_imgs"].float(),
        sample_cuda["ref_intrinsics"].float(),
        sample_cuda["src_intrinsics"].float(),
        sample_cuda["ref_extrinsics"].float(),
        sample_cuda["src_extrinsics"].float(),
        sample_cuda["depth_min"].float(),
        sample_cuda["depth_max"].float()
    )

    depth_est_list = outputs["depth_est_list"]
    scalar_outputs = {}
    image_outputs = {}
    image_outputs["depth_gt"] = ref_depths[:, 0]
    image_outputs["ref_img"] = sample["imgs"][:, 0]
    image_outputs["mask"] = (ref_depths[:, 0] > 0).float()

    dHeight = ref_depths.shape[2]
    dWidth = ref_depths.shape[3]
    standard_loss = []
    #segment_loss = []
    for i in range(0, args.nscale):
        depth_gt = ref_depths[:, 0, :, :]
        mask = depth_gt > 0
        mask = mask.float()
        depth_est = depth_est_list[i]
        depth_est = F.interpolate(depth_est.unsqueeze(1), size=[dHeight, dWidth]).squeeze(1)

        standard_loss.append(criterion(sample_cuda["imgs"], sample_cuda["cams"], depth_est))

        #segment_loss_t, ref_seg, view_segs = criterion_seg(sample_cuda["imgs_seg"], sample["cams"].cuda(), depth_est)
        #segment_loss.append(torch.mean(segment_loss_t) * args.w_seg)

        scalar_outputs["reconstr_loss_{}".format(i)] = criterion.reconstr_loss
        scalar_outputs["ssim_loss_{}".format(i)] = criterion.ssim_loss
        scalar_outputs["smooth_loss_{}".format(i)] = criterion.smooth_loss
        image_outputs["depth_est_{}".format(i)] = depth_est * mask
        '''
        if i == 0:
            # ref_seg: [B, H, W, C]
            ref_seg_idx = torch.argmax(ref_seg, dim=3)  # [B, H, W]
            ref_seg_viz = seg_viz.convert_heatmap(ref_seg_idx.cpu().numpy())  # [B, H, W, C]
            image_outputs["ref_seg"] = ref_seg_viz

            for j in range(view_segs.size(1)):
                view_seg = view_segs[:, j]
                view_seg_idx = torch.argmax(view_seg, dim=3)  # [B, H, W]
                view_seg_viz = seg_viz.convert_heatmap(view_seg_idx.cpu().numpy())  # [B, H, W, C]
                image_outputs["src_img_{}".format(j + 1)] = sample["imgs"][:, 1 + j]
                image_outputs["src_seg_{}".format(j + 1)] = view_seg_viz
        '''
    standard_loss = sum(standard_loss)
    #segment_loss = sum(segment_loss)
    scalar_outputs["standard_loss"] = standard_loss
    #scalar_outputs["segment_loss"] = segment_loss
    loss = standard_loss #+ segment_loss

    loss.backward()  # 反向传播计算得到每个参数的梯度值

    optimizer.step()  # 通过梯度下降执行一步参数更新（optimizer.step()）

    scalar_outputs["loss"] = loss
    # scalar_outputs={"loss":loss}

    #if detailed_summary:
    depth_est = depth_est_list[0].float()
    depth_gt = ref_depths[:, 0, :, :].float()
    mask = depth_gt > 0
    mask = mask.float()
    depth_interval = sample_cuda["cams"][:, 0, 1, 3, 1]
    image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)
    scalar_outputs["mae"] = non_zero_mean_absolute_diff(depth_gt, depth_est, depth_interval)
    scalar_outputs["less_one_accuracy"] = less_one_percentage(depth_gt, depth_est, depth_interval)
    scalar_outputs["less_three_accuracy"] = less_three_percentage(depth_gt, depth_est, depth_interval)

    scalar_outputs = reduce_scalar_outputs(scalar_outputs)
    return tensor2float(loss), tensor2float(scalar_outputs), tensor2numpy(image_outputs), tensor2float(standard_loss), depth_est_list

'''
def train_sample_aug(sample, depth_est_list, detailed_summary=False):
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    ref_depths = sample_cuda["ref_depths"]
    ref_img_aug = sample_cuda["ref_img_aug"]
    src_imgs_aug = sample_cuda["src_imgs_aug"]
    ref_img_aug, filter_mask = random_image_mask(ref_img_aug,
                                                 filter_size=(ref_img_aug.size(2) // 3, ref_img_aug.size(3) // 3))

    outputs = model(
        ref_img_aug.float(),
        src_imgs_aug.float(),
        sample_cuda["ref_intrinsics"].float(),
        sample_cuda["src_intrinsics"].float(),
        sample_cuda["ref_extrinsics"].float(),
        sample_cuda["src_extrinsics"].float(),
        sample_cuda["depth_min"].float(),
        sample_cuda["depth_max"].float()
    )

    depth_est_list_aug = outputs["depth_est_list"]
    scalar_outputs = {}
    image_outputs = {}
    filter_mask = filter_mask[:, 0, :, :]

    dHeight = ref_depths.shape[2]
    dWidth = ref_depths.shape[3]
    augment_loss = []
    for i in range(0, args.nscale):
        depth_gt = ref_depths[:, 0, :, :]
        mask = depth_gt > 0
        mask = mask.float()
        depth_est = depth_est_list[i].clone().detach()
        depth_est = F.interpolate(depth_est.unsqueeze(1), size=[dHeight, dWidth]).squeeze(1)
        depth_est_aug = depth_est_list_aug[i]
        depth_est_aug = F.interpolate(depth_est_aug.unsqueeze(1), size=[dHeight, dWidth]).squeeze(1)
        augment_loss.append(criterion_aug(depth_est_aug, depth_est, filter_mask) * args.w_aug)
        image_outputs["depth_est_aug_{}".format(i)] = depth_est * mask

    augment_loss = sum(augment_loss)

    augment_loss.backward()

    optimizer.step()

    scalar_outputs["augment_loss"] = augment_loss

    scalar_outputs = reduce_scalar_outputs(scalar_outputs)
    return tensor2float(augment_loss), tensor2float(scalar_outputs), tensor2numpy(image_outputs)
'''

if __name__ == '__main__':
    if args.mode == "train":
        train()
        dist.destroy_process_group()  # 销毁进程
