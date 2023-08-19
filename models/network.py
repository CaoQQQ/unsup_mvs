# -*- coding: utf-8 -*-
# @Time    : 2020/04/16 15:42
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : network
# @Software: PyCharm

import torch
import torch.nn as nn
from models.modules import *
from torch.nn import Transformer

'''
一、self参数
self指的是实例Instance本身，在Python类中规定，函数的第一个参数是实例对象本身，并且约定俗成，把其名字写为self，也就是说，类中的方法的第一个参数一定要是self，而且不能省略。
二、__ init__ ()方法
在python中创建类后，通常会创建一个\ __ init__ ()方法，这个方法会在创建类的实例的时候自动执行。 \ __ init__ ()方法必须包含一个self参数，而且要是第一个参数。
三、super(Net, self).__init__()
Python中的super(Net, self).__init__()是指首先找到Net的父类（比如是类NNet），然后把类Net的对象self转换为类NNet的对象，然后“被转换”的类NNet对象调用自己的init函数，
其实简单理解就是子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()的那些东西。
'''


# Feature pyramid
class FeaturePyramid(nn.Module):
    def __init__(self):
        super(FeaturePyramid, self).__init__()
        # 下面表示该卷积网络有几个层 每个层具体的参数是什么
        # 还未构建金字塔网络，只是说明这个类的一些重要参数
        # conv(输入通道数量，输出通道数量，卷积核大小，步长）
        self.conv0aa = conv(3,  64, kernel_size=3, stride=1)
        self.conv0ba = conv(64, 64, kernel_size=3, stride=1)
        self.conv0bb = conv(64, 64, kernel_size=3, stride=1)
        self.conv0bc = conv(64, 32, kernel_size=3, stride=1)
        self.conv0bd = conv(32, 32, kernel_size=3, stride=1)
        self.conv0be = conv(32, 32, kernel_size=3, stride=1)
        self.conv0bf = conv(32, 16, kernel_size=3, stride=1)
        self.conv0bg = conv(16, 16, kernel_size=3, stride=1)
        self.conv0bh = conv(16, 16, kernel_size=3, stride=1)
        '''
        self.conv0a01 = conv(3, 16, kernel_size=3, stride=1)
        self.conv0a12 = conv(16, 32, kernel_size=3, stride=1)
        self.conv0a23 = conv(32, 64, kernel_size=3, stride=1)
        self.conv0ab = conv(64, 64, kernel_size=3, stride=1)
        self.conv0b12 = conv(64, 32, kernel_size=3, stride=1)
        self.conv0b23 = conv(32, 16, kernel_size=3, stride=1)
        self.conv0c12 = conv(64, 32, kernel_size=3, stride=1)
        self.conv0c23 = conv(32, 16, kernel_size=3, stride=1)
        self.conv0out = conv(16, 16, kernel_size=3, stride=1)
        '''
        #self.downsample1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1)
        #self.downsample2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)

        # 注意力机制定义
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        # 初始化权重
        self._init_weights()

    def forward(self, img, scales=5):
        # 金字塔层数默认为5，但是训练中可以自己设置
        # fp用来存储特征图，它是list类型的
        fp = []
        '''
        fa1_16 = self.conv0a01(img)
        fa2_32 = self.conv0a12(fa1_16)
        fa3_64 = self.conv0a23(fa2_32)
        fb1_64 = self.conv0ab(fa3_64)
        fb2_32 = self.conv0b12(fb1_64)
        fb3_16 = self.conv0b23(fb2_32)
        fb2_cat_64 = torch.cat([fa2_32, fb2_32], dim=1)
        fb3_cat_32 = torch.cat([fa1_16, fb3_16], dim=1)
        fc1_64 = fb1_64 + fb2_cat_64
        fc2_32 = self.conv0c12(fc1_64) + fb3_cat_32
        fc3_16 = self.conv0c23(fc2_32)
        f_out = self.conv0out(fc3_16)

        '''
        # 首先进行了九层卷积，从3通道变成了16通道的特征图，图像分辨率不变

        f = self.conv0aa(img)
        f_out = self.conv0bh(
            self.conv0bg(self.conv0bf(self.conv0be(self.conv0bd(self.conv0bc(self.conv0bb(self.conv0ba(f))))))))
        '''
        f1_64 = self.conv0aa(img)
        f2_64 = self.conv0ba(f1_64)
        f3_64 = self.conv0bb(f2_64) + f1_64
        f4_32 = self.conv0bc(f3_64)
        f5_32 = self.conv0bd(f4_32)
        f6_32 = self.conv0be(f5_32) + f4_32
        f7_16 = self.conv0bf(f6_32)
        f8_16 = self.conv0bg(f7_16)
        f9_16 = self.conv0bh(f8_16) + f7_16
        '''
        # 把原图大小的特征图存储
        attention_weights = self.attention(f_out)
        f_att = f_out * attention_weights
        fp.append(f_att)
        # 循环下采样输入图片
        # 注意原图算金字塔底层 因此scales要-1
        for scale in range(scales - 1):
            # 下采样函数
            '''
            img = self.downsample(img)  # 使用卷积来实现下采样
            fa1_16 = self.conv0a01(img)
            fa2_32 = self.conv0a12(fa1_16)
            fa3_64 = self.conv0a23(fa2_32)
            fb1_64 = self.conv0ab(fa3_64)
            fb2_32 = self.conv0b12(fb1_64)
            fb3_16 = self.conv0b23(fb2_32)
            fb2_cat_64 = torch.cat([fa2_32, fb2_32], dim=1)
            fb3_cat_32 = torch.cat([fa1_16, fb3_16], dim=1)
            fc1_64 = fb1_64 + fb2_cat_64
            fc2_32 = self.conv0c12(fc1_64) + fb3_cat_32
            fc3_16 = self.conv0c23(fc2_32)
            f_out = self.conv0out(fc3_16)

            '''
            # 首先进行了九层卷积，从3通道变成了16通道的特征图，图像分辨率不变
            #img = self.downsample1(img)  # 使用卷积来实现下采样
            img = nn.functional.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=None).detach()
            f = self.conv0aa(img)
            f_out = self.conv0bh(
                self.conv0bg(self.conv0bf(self.conv0be(self.conv0bd(self.conv0bc(self.conv0bb(self.conv0ba(f))))))))

            # 把原图大小的特征图存储
            attention_weights = self.attention(f_out)
            f_att = f_out * attention_weights
            #fp.append(f_att)
            prev_f = fp[-1]
            prev_f = nn.functional.interpolate(prev_f, size=f_att.size()[2:], mode='bilinear', align_corners=False)
            f_add = f_att + prev_f
            fp.append(f_add)
        # 返回一个list，里面包含了不同尺度的特征图，即不同尺寸的tensor
        return fp


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)



# 代价体正则化网络 一个三维卷积网络
class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        # 如果希望卷积后大小跟原来一样，则需要设置padding=(kernel_size-1)/2=1。
        '''
        self.conv0  = SelfAttentionConvBnReLU3D(16, 16, kernel_size=3, pad=1)
        self.conv0a = SelfAttentionConvBnReLU3D(16, 16, kernel_size=3, pad=1)
        self.conv1  = SelfAttentionConvBnReLU3D(16, 32, stride=2, kernel_size=3, pad=1)# 此处stride=2，图片大小会缩小一半
        self.conv2  = SelfAttentionConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        self.conv2a = SelfAttentionConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        self.conv3  = SelfAttentionConvBnReLU3D(32, 64, kernel_size=3, pad=1)
        self.conv4  = SelfAttentionConvBnReLU3D(64, 64, kernel_size=3, pad=1)
        self.conv4a = SelfAttentionConvBnReLU3D(64, 64, kernel_size=3, pad=1)
        '''
        self.conv0 = ConvBnReLU3D(16, 16, kernel_size=3, pad=1)
        self.conv0a = ConvBnReLU3D(16, 16, kernel_size=3, pad=1)
        self.conv1 = ConvBnReLU3D(16, 32, stride=2, kernel_size=3, pad=1)  # 此处stride=2，图片大小会缩小一半
        self.conv2 = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        self.conv2a = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        self.conv3 = ConvBnReLU3D(32, 64, kernel_size=3, pad=1)
        self.conv4 = ConvBnReLU3D(64, 64, kernel_size=3, pad=1)
        self.conv4a = ConvBnReLU3D(64, 64, kernel_size=3, pad=1)
        # nn.Sequential是序列容器
        # 与一层一层的单独调用模块组成序列相比，nn.Sequential() 可以允许将整个容器视为单个模块
        # 相当于把多个模块封装成一个模块
        # forward()方法接收输入之后，nn.Sequential()按照内部模块的顺序自动依次计算并输出结果。
        # nn.Sequential()内的网络模块之间是按照添加的顺序级联的
        self.conv5 = nn.Sequential(
            # nn.ConvTranspose3d
            # output_padding ( int or tuple , optional ) – 附加大小添加到输出形状中每个维度的一侧。默认值：0
            # bias(bool, optional) - - 如果为True ，则向输出添加可学习的偏差。默认值： True
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=0, stride=1, bias=False),
            # 每个batch的数据规范化为统一的分布 这里的输入是N C D H W
            nn.BatchNorm3d(32),
            # inplace = True ,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
            # Relu函数作用：
            # if input > 0:
            # return input
            # else:
            # return 0
            # ReLU的梯度只可以取两个值：0或1，
            # 当输入小于0时，梯度为0；当输入大于0时，梯度为1。
            # 好处就是：ReLU的梯度的连乘不会收敛到0 ，连乘的结果也只可以取两个值：0或1 ，
            # 如果值为1 ，梯度保持值不变进行前向传播；如果值为0 ,梯度从该位置停止前向传播。
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            # 每个batch的数据规范化为统一的分布
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        '''
        self.transformer = Transformer(
            d_model=8 * 8 * 8,  # 输入的特征维度
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.1
        )
        '''
        self.prob0 = nn.Conv3d(16, 1, 3, stride=1, padding=1)

        # 初始化权重
        self._init_weights()

    # U-net结构的卷积网络
    def forward(self, x):
        # 通过以下卷积操作，将16通道的三维代价体变成64通道的三维代价体，代价体大小不变
        conv0 = self.conv0a(self.conv0(x))  # conv0还是原来大小,C=16
        # conv2和conv4的代价体大小皆为原来的一半
        conv2 = self.conv2a(self.conv2(self.conv1(conv0)))  # C=32
        conv4 = self.conv4a(self.conv4(self.conv3(conv2)))  # C=64
        # self.conv5进行了反卷积，但是反卷积后的大小仍不变，仅仅将64通道变为32通道
        conv5 = conv2 + self.conv5(conv4)
        # self.conv6进行了反卷积，但是反卷积后的大小增大了一倍==原代价体一样大小
        conv6 = conv0 + self.conv6(conv5)
        '''
        batch_size, channels, depth, height, width = conv6.shape
        transformer_input = conv6.view(batch_size, channels, -1)  # 将空间维度合并成一个维度
        transformer_input = transformer_input.permute(2, 0, 1)  # 将特征维度放在最前面
        transformer_output = self.transformer(transformer_input)
        transformer_output = transformer_output.permute(1, 2, 0)  # 恢复原始维度顺序
        transformer_output = transformer_output.view(batch_size, channels, depth, height, width)
        # 下面这个步骤将16通道的代价体卷积为1通道的，并且squeeze(1)去除了channel维度
        prob = self.prob0(transformer_output).squeeze(1)
        '''
        prob = self.prob0(conv6).squeeze(1)
        return prob

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CVPMVSNet(nn.Module):
    def __init__(self, args):
        # args是传入的参数
        super(CVPMVSNet, self).__init__()
        # 金字塔网络参数 forward(self, img, scales=5):
        self.featurePyramid = FeaturePyramid()
        # 代价体网络参数 forward(self, x) x是输入的代价体
        self.cost_reg_refine = CostRegNet()
        # train文件中的参数
        self.args = args

    # foward中表示的是要传入的参数 在前面处理时，所有的数据前面都要加上Batch维度，这个和dtu文件里面是不一样的
    # 在utils.py:
    # depth_min = float(lines[11].split()[0])
    # depth_interval = float(lines[11].split()[1])
    # depth_max = depth_min+(256*depth_interval)
    def forward(self, ref_img, src_imgs, ref_in, src_in, ref_ex, src_ex, depth_min, depth_max):
        ## Initialize output list for loss
        # 初始化loss的输出列表
        depth_est_list = []
        output = {}

        # ref_img: [batch_size, 3, height, width] 这里的注释指的是大小
        # ref_img: [batch_size, nsrc, 3, height, width]

        ## Feature extraction
        ref_feature_pyramid = self.featurePyramid(ref_img, self.args.nscale)  # 输出参考图像不同尺度的特征图list
        src_feature_pyramids = []  # 源图像list初始化
        # 对于每个源图像
        for i in range(self.args.nsrc):
            # src_imgs的尺寸：Batch×nsrc×C×H×W
            # 这里是针对每一个源图像循环输出源图像特征金字塔list，也就是源图像list里面包含了特征list
            src_feature_pyramids.append(self.featurePyramid(src_imgs[:, i, :, :, :], self.args.nscale))

        # ref_feature_pyramid: [batich_size, channels, height, wdth]
        # src_feature_pyramids: ([batch_size, channels, height, width], ...)  省略号表示有nsrc个b c h w尺寸的tensor矩阵

        # Pre-conditioning corresponding multi-scale intrinsics for the feature:
        # 根据特征金字塔的缩放构建相机内参金字塔(对于每个视角相机外参都是一样的) 形参分别是参考图像内参矩阵，参考图像尺寸以及每个特征图的大小形成的list
        # conditionIntrinsics函数返回值： [B, nScale, 3, 3]
        # 这里的循环中因为feature是tensor类型的因此可以直接用shape函数
        ref_in_multiscales = conditionIntrinsics(ref_in, ref_img.shape, [feature.shape for feature in
                                                                         ref_feature_pyramid])  # [B, nScale, 3, 3]
        src_in_multiscales = []
        for i in range(self.args.nsrc):
            src_in_multiscales.append(conditionIntrinsics(src_in[:, i], ref_img.shape,
                                                          [feature.shape for feature in src_feature_pyramids[i]]))
        # 因为src_in_multiscales的格式为 nsrc B nscale 3 3，因此此处将B提到最前面
        src_in_multiscales = torch.stack(src_in_multiscales).permute(1, 0, 2, 3, 4)  # B nsrc nscale 3 3

        ## 估计初始深度图:
        # 初始深度范围假设
        # 构建ref volume
        # homo构建warp volume
        # 通过方差聚合得到cost volume
        # 通过代价体正则化网络得到cost_reg
        # 再通过softmax得到prob volume
        # 概率体和深度假设求乘积回归得到原始深度图

        ## Estimate initial coarse depth map
        # 计算最粗糙的深度假设，深度假设层D=48，从425～1065共采样48层
        # calSweepingDepthHypo会生成目标范围内的初始48层实际深度值，根据内参文件而来
        depth_hypos = calSweepingDepthHypo(ref_in_multiscales[:, -1], src_in_multiscales[:, 0, -1], ref_ex, src_ex,
                                           depth_min, depth_max)
        # depth_hypos: [batch_size, num_depths] 此处D=48,且已经返回粗糙的深度值

        # step 2. differentiable homograph, build cost volume
        # list[-1]：返回最后一个数据 此处ref_feature_pyramid[-1]的作用为拿出金字塔最顶层的特征图
        # ref_feature_pyramid是一个列表，其中每个tensor大小为：[batich_size, channels, height, wdth]
        ref_volume = ref_feature_pyramid[-1].unsqueeze(2).repeat(1, 1, len(depth_hypos[0]), 1, 1)
        volume_sum = ref_volume  # ref_volume：B,C,NUMDEP,H,W。depth=48
        # 相比于论文里面计算代价体方差公式，代码里面做了简化计算：
        # C=(Σ(Vi)^2)/N-(Vmean)^2
        # 做方差是为了适应输入图像数量的不同
        volume_sq_sum = ref_volume.pow_(2)  # pow函数让tensor矩阵逐元素点乘，pow_()会让结果替代原变量，即ref_volume的值会改变
        if self.args.mode == "test" or self.args.mode == "eval":
            del ref_volume
        for src_idx in range(self.args.nsrc):
            # warpped features
            # 对于每一个源图像都要计算其代价体
            # src_feature_pyramids: ([batch_size, channels, height, width], ...)  省略号表示有nsrc个b c h w尺寸的tensor矩阵
            # src_feature_pyramids[src_idx][-1]表示取每个源图像最上面的那层特征图矩阵
            # ref_in_multiscales表示缩放后的参考图像内参，其维度为# [B, nScale, 3, 3]。ref_in_multiscales[:, -1]：[B,3,3],这里的-1取了最上面的那层
            # src_in_multiscales表示缩放后的源图像内参 其维度为 [B nsrc nscale 3 3]。src_in_multiscales[:, src_idx, -1, :, :]与上同理
            # ref_ex 参考图像外参 [B 4 4]
            # src_ex 源图像外参 [B,nsrc,4,4] 这一部分详见dtu文件中的read_cam_file函数
            warped_volume = homo_warping(src_feature_pyramids[src_idx][-1], ref_in_multiscales[:, -1],
                                         src_in_multiscales[:, src_idx, -1, :, :], ref_ex, src_ex[:, src_idx],
                                         depth_hypos)
            # 将每个warped_volume与ref_volume相加
            if self.args.mode == "train":
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            elif self.args.mode == "test" or self.args.mode == "eval":
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
                del warped_volume
            else:
                print("Wrong!")
        # ref_volume: [batch_size, channels, num_depth, height, width]
        # warped_volume: [batch_size, channels, num_depth, height, width]

        # Aggregate multiple feature volumes by variance
        # 加_表示替代原值
        # div表示元素点除，sub表示减，pow表示元素点乘
        # volume_sq_sum 表示V^2 ；volume_sum表示V
        cost_volume = volume_sq_sum.div_(self.args.nsrc + 1).sub_(volume_sum.div_(self.args.nsrc + 1).pow_(2))
        if self.args.mode == "test" or self.args.mode == "eval":
            del volume_sum
            del volume_sq_sum
        # cost_volume: [batch_size, channels, num_depth, height, width]

        # Regularize cost volume
        # 通过代价体正则化网络得到cost_reg
        # 输入：[batch_size, channel, numdepth=48, height, width]
        cost_reg = self.cost_reg_refine(cost_volume)
        # cost_reg: [batch_size, num_depth, height, width]

        # 通过Softmax函数，将深度维度的信息压缩为0～1之间的分布，得到概率体probability volume
        prob_volume = F.softmax(cost_reg, dim=1)  # prob_volume: [batch_size, num_depth, height, width]
        # 通过深度回归depth regression，得到估计的最优深度图 [B, H, W]
        depth = depth_regression(prob_volume, depth_values=depth_hypos)
        # 添加初始深度图到list
        depth_est_list.append(depth)
        # depth: [batch_size, height, width]

        ##上采样深度图并通过特征金字塔refine
        # 两倍上采样深度图
        # 计算深度假设范围
        # 计算cost volume
        # 通过代价体正则化网络得到cost_reg
        # 再通过softmax得到prob volume
        # 概率体和深度假设回归得到更精细的深度图
        # 把所有得到的深度图都放到数组里

        ## Upsample depth map and refine along feature pyramid
        # 上面这些步骤得到了初始深度图（金字塔最上层深度图）。 最上层代价体以及一些金字塔特征图
        # train阶段scale为2，因此range(0,-1,-1)只有[0]。test阶段，scale为5，range(3,-1,-1)=[3,2,1,0]。
        for level in range(self.args.nscale - 2, -1, -1):  # 计数到 stop 结束，但不包括 stop。例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
            ## Upsample
            # 类比卷积中的下采样操作：
            # img = nn.functional.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=None).detach()
            # depth[None, :]中None起升维的作用，即在Depth[B H W]的前面增加一个维度
            ## nn.functional.interpolate注意点：
            # 输入的张量数组里面的数据类型必须是float。
            # 输入的数组维数只能是3、4或5，分别对应于时间、空间、体积采样。
            # 不对输入数组的前两个维度(批次和通道)采样，从第三个维度往后开始采样处理。
            # 输入的维度形式为：批量(batch_size)×通道(channel)×[可选深度]×[可选高度]×宽度(前两个维度具有特殊的含义，不进行采样处理)
            # size与scale_factor两个参数只能定义一个，即两种采样模式只能用一个。要么让数组放大成特定大小、要么给定特定系数，来等比放大数组。
            # 如果size或者scale_factor输入序列，则必须匹配输入的大小。如果输入四维，则它们的序列长度必须是2，如果输入是五维，则它们的序列长度必须是3。
            # mode是’linear’时输入必须是3维的；是’bicubic’时输入必须是4维的；是’trilinear’时输入必须是5维的
            # 如果align_corners被赋值，则mode必须是'linear'，'bilinear'，'bicubic'或'trilinear'中的一个。

            depth_up = nn.functional.interpolate(depth[None, :], size=None, scale_factor=2, mode='bilinear',
                                                 align_corners=None)  # 上采样到上一层，深度图变大
            depth_up = depth_up.squeeze(0)  # 去除前面增加的维度 [B H W]
            # Generate depth hypothesis
            depth_hypos = calDepthHypo(self.args, depth_up, ref_in_multiscales[:, level, :, :],
                                       src_in_multiscales[:, :, level, :, :], ref_ex, src_ex, depth_min, depth_max,
                                       level)

            cost_volume = proj_cost(self.args, ref_feature_pyramid[level], src_feature_pyramids, level,
                                    ref_in_multiscales[:, level, :, :], src_in_multiscales[:, :, level, :, :], ref_ex,
                                    src_ex[:, :], depth_hypos)

            cost_reg2 = self.cost_reg_refine(cost_volume)
            if self.args.mode == "test" or self.args.mode == "eval":
                del cost_volume

            prob_volume = F.softmax(cost_reg2, dim=1)
            if self.args.mode == "test" or self.args.mode == "eval":
                del cost_reg2

            # Depth regression
            depth = depth_regression_refine(prob_volume, depth_hypos)

            depth_est_list.append(depth)

        # Photometric confidence
        with torch.no_grad():
            num_depth = prob_volume.shape[1]
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                                stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device,
                                                                                  dtype=torch.float)).long()
            prob_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        if self.args.mode == "test" or self.args.mode == "eval":
            del prob_volume

        ## Return
        depth_est_list.reverse()  # Reverse the list so that depth_est_list[0] is the largest scale.
        output["depth_est_list"] = depth_est_list
        output["prob_confidence"] = prob_confidence

        return output


def sL1_loss(depth_est, depth_gt, mask):
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')


def MSE_loss(depth_est, depth_gt, mask):
    return F.mse_loss(depth_est[mask], depth_gt[mask], size_average=True)


def gradient_x(img):
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gx


def gradient_y(img):
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gy


def get_disparity_smoothness(disp, image):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)
    image_gradients_x = gradient_x(image)
    image_gradients_y = gradient_y(image)
    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))
    # print('weights_x: {}'.format(weights_x.shape))
    # print('weights_y: {}'.format(weights_y.shape))
    # print('disp_gradients_x: {}'.format(disp_gradients_x.shape))
    # print('disp_gradients_y: {}'.format(disp_gradients_y.shape))
    smoothness_x = [disp_gradients_x * weights_x]
    smoothness_y = [disp_gradients_y * weights_y]
    return smoothness_x + smoothness_y


def get_valid_map(disp):
    mask = torch.where(disp > 0, torch.ones_like(disp, device=disp.device),
                       torch.zeros_like(disp, device=disp.device))
    return mask


def gradient_smoothness_loss(depth_est, depth_gt, ref_img):
    depth_est = depth_est.unsqueeze(dim=1)
    mask = get_valid_map(depth_est)
    # print('mask: {}'.format(mask.shape))
    mask_list = [mask[:, :, :, 1:], mask[:, :, 1:, :]]
    disp_smoothness = get_disparity_smoothness(depth_est, ref_img)
    loss = [torch.mean(torch.mul(mask_list[i], torch.abs(disp_smoothness[i]))) for i in range(2)]
    loss = sum(loss)
    return loss
