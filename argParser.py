# -*- coding: utf-8 -*-
# @Time    : 2020/04/16 16:59
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : argParser
# @Software: PyCharm

import argparse


def getArgsParser():
    parser = argparse.ArgumentParser(description='Cost Volume Pyramid Based Depth Inference for Multi-View Stereo')

    # General settings
    # 导入目前训练的信息
    parser.add_argument('--info', default='None', help='Info about current run')
    parser.add_argument('--log', default='None', help='Info about current run')
    # 设置训练模式
    parser.add_argument('--mode', default='train', help='train or test ro validation', choices=['train', 'test', 'eval'])

    # Data settings
    # 深度假设数量，一共假设这么多种不同的深度，在里面找某个像素的最优深度
    parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
    # trainlist无用
    parser.add_argument('--trainlist', default=None, help='train list')
    # 选择训练集，之后可以选择ETH3D或者blendMVS或者tank and temple
    parser.add_argument('--dataset', default='dtu', help='select dataset')
    # 数据集在电脑中的位置
    parser.add_argument('--dataset_root', help='path to dataset root')
    # 选择输入图片的高度 其中128是训练集 1200是测试集
    parser.add_argument('--imgsize', type=int, default=128, choices=[128, 1200], help='height of input image')
    # 输入源图像的数量 1张参考图像+2张源图像
    parser.add_argument('--nsrc', type=int, default=2, help='number of src views to use')
    # 金字塔层数，图片分辨率小层数就小
    parser.add_argument('--nscale', type=int, default=5, help='number of scales to use')

    # Training settings
    parser.add_argument('--epochs', type=int, default=28, help='number of epochs to train')
    # 初始学习率
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    # lrepochs: 训练中采用了动态调整学习率的策略，在第10，12，14轮训练的时候，让learning_rate除以2变为更小的学习率
    parser.add_argument('--lrepochs', type=str, default="10,12,14,20:2",
                        help='epoch ids to downscale lr and the downscale rate')
    # wd: weight_decay（权重衰退），作为Adam优化器超参数，实现中并未使用。
    # 使用 weight decay 可以：防止过拟合和保持权重在一个较小在的值，避免梯度爆炸。
    # 在深度学习模型中,一般将衰减系数设置为 `0.0001` 到 `0.001` 之 间的值
    # 论文里是验证了1e-4比较好
    # 当你不确定模型复杂度和数据集大小的时候，最保守就是从`1e-4`周围开始尝试
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    # BATCH_SIZE: 即一次训练所抓取的数据样本数量；
    # 在训练过程中，可以将batch_size作为超参数做多次尝试。
    # 另一方面，也可以在前期使用较大的学习率和较越BatchSize粗调，后期（比如论文实验/比赛最后）将BatchSize变小精调，并使用SGD优化方法，慢慢把Error磨低。
    parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
    # 记录的迭代间隔，即多少次迭代后打印一次
    parser.add_argument('--summary_freq', type=int, default=200, help='print and summary frequency')
    # 保存的checkpoint间隔
    parser.add_argument('--save_freq', type=int, default=2000, help='save checkpoint frequency')
    # 随机数种子
    # 深度学习网络模型中初始的权值参数通常都是初始化成随机数，为了能够完全复现实验结果，需要固定随机种子。即产生随机种子意味着每次运行实验，产生的随机数都是相同的。默认为1。
    #parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    # 没用到
    parser.add_argument('--loss_function', default='sl1', help='which loss function to use', choices=['sl1', 'mse'])

    # Checkpoint settings
    parser.add_argument('--loadckpt', type=str, default='', help='load a specific checkpoint')
    parser.add_argument('--logckptdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
    parser.add_argument('--loggingdir', default='./logs/', help='the directory to save logging outputs')
    # 是否继续训练
    parser.add_argument('--resume', type=int, default=0, help='continue to train the model')

    # Evaluation settings
    parser.add_argument('--outdir', default='./outputs/debug/', help='the directory to save depth outputs')
    parser.add_argument('--eval_visualizeDepth', type=int, default=1)
    parser.add_argument('--eval_prob_filtering', type=int, default=0)
    parser.add_argument('--eval_prob_threshold', type=float, default=0.99)
    parser.add_argument('--eval_shuffle', type=int, default=0)

    parser.add_argument('--interval_scale', type=float, default=1.06)
    parser.add_argument('--summarydir', type=str, default='summary')

    #parser.add_argument('--seg_clusters', type=int, default=4, help='cluster centers for unsupervised co-segmentation')
    #parser.add_argument('--w_seg', type=float, default=0.01, help='weight for segments reprojection loss')
    #parser.add_argument('--w_aug', type=float, default=0.01, help='weight of augment loss')

    #parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')
    parser.add_argument("--local_rank", type=int, default=0)  # 多GPU分布训练所用参数
    return parser


def checkArgs(args):
    # Check if the settings is valid
    # 检测设置的mode是否是train eval test中的一种
    # Python assert（断言）用于判断一个表达式，在表达式条件为false的时候触发异常。
    assert args.mode in ["train", "eval", "test"]
    # 下面的两则断言根本没有用 因为args.loadckpt始终为空，因此断言始终为true，具体的ckpt导入路径需要在代码里自己编辑，具体可查询sw_path
    if args.resume:
        # 如果resume为1
        # 检查路径是否为空
        assert len(args.loadckpt) == 0
    if args.loadckpt:
        assert args.resume is 0