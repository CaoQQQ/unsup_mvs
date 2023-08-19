# -*- coding: utf-8 -*-
# @Time    : 2020/04/16 14:31
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com or 17770026885@163.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : modules
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


# 缩放内参
def conditionIntrinsics(intrinsics, img_shape, fp_shapes):
    # Pre-condition intrinsics according to feature pyramid shape.

    # Calculate downsample ratio for each level of feture pyramid
    down_ratios = []
    for fp_shape in fp_shapes:
        # 计算下采样的比例，这里的tensor是三维的
        down_ratios.append(img_shape[2] / fp_shape[2])# 得到了金字塔下采样的比例list

    # condition intrinsics
    intrinsics_out = []
    for down_ratio in down_ratios:
        intrinsics_tmp = intrinsics.clone()
        intrinsics_tmp[:, :2, :] = intrinsics_tmp[:, :2, :] / down_ratio
        intrinsics_out.append(intrinsics_tmp)
    # 由于intrinsics_tmp的数据维度是B 3 3，因此 intrinsics_out的数据维度是 nscale B 3 3,需要调换位置
    return torch.stack(intrinsics_out).permute(1, 0, 2, 3)  # [B, nScale, 3, 3]

# 没有用到
def calInitDepthInterval(ref_in, src_in, ref_ex, src_ex, pixel_interval):
    return 165  # The mean depth interval calculated on 4-1 interval setting...

# depth_hypos = calSweepingDepthHypo
# (ref_in_multiscales[:, -1], src_in_multiscales[:, 0, -1],
# ref_ex, src_ex,depth_min, depth_max)
def calSweepingDepthHypo(ref_in, src_in, ref_ex, src_ex,
                         depth_min, depth_max, nhypothesis_init=48):
    # Batch
    batchSize = ref_in.shape[0]
    # print('depth_max: {}'.format(depth_max))
    # print('depth_min: {}'.format(depth_min))
    depth_range = depth_max[0] - depth_min[0]
    depth_interval_mean = depth_range / (nhypothesis_init - 1)
    # Make sure the number of depth hypothesis has a factor of 2
    assert nhypothesis_init % 2 == 0
    # unsqueeze()函数起升维的作用,参数表示在哪个地方加一个维度。
    # range:根据步长创建一维tensor
    # torch.range(start=0, end, step=1)
    depth_hypos = torch.range(depth_min[0], depth_max[0], depth_interval_mean).unsqueeze(0)
    # Assume depth range is consistent in one batch.
    for b in range(1, batchSize):
        # depth_range = depth_max[b] - depth_min[b]
        # 注意torch.stack堆叠时会增加一个维度，默认在最前面增加一个维度；
        # torch.cat堆叠时不会增加一个维度，可以理解为两个矩阵在指定维度相融合，例如：
        # a=[1,1] b=[2,2]
        # c=torch.stack((a,b))=[[1,1],[2,2]] shape:2×1×2
        # c=torch.cat((a,b))=[1,1;2,2] shape:2×2
        depth_hypos = torch.cat(
            (depth_hypos, torch.range(depth_min[0], depth_max[0], depth_interval_mean).unsqueeze(0)), 0)
    return depth_hypos.cuda()# 这里为什么要把数据放到GPU上啊 模型不是已经加载到GPU上了吗


def homo_warping(src_feature, ref_in, src_in, ref_ex, src_ex, depth_hypos):
    # Apply homography warpping on one src feature map from src to ref view.
    # src_feature_pyramids[src_idx][-1]表示取每个源图像最上面的那层特征图矩阵
    # ref_in：[B,3,3],这里的-1取了最上面的那层
    # src_in的维度同上
    # ref_ex 参考图像外参 [B 4 4]
    # src_ex 源图像外参 [B,4,4]
    ##extrinsic：
    #0.802256 -0.439347 0.404178 -291.419
    #0.427993 0.895282 0.123659 -77.0495
    #-0.416183 0.073779 0.906283 71.2762
    #0.0 0.0 0.0 1.0
    ##intrinsic：
    #361.54125 0.0 82.9005
    #0.0 360.3975 66.383625
    #0.0 0.0 1.0
    ##425.0 2.5

    batch, channels = src_feature.shape[0], src_feature.shape[1]
    num_depth = depth_hypos.shape[1]
    height, width = src_feature.shape[2], src_feature.shape[3]

    with torch.no_grad():# 阻止梯度计算，降低计算量，保护数据
        # 内参矩阵乘上外参矩阵计算单应性矩阵
        # src_ex[:, 0:3, :]此处将[B 4 4]变成了[B 3 4] 方便和内参矩阵相乘
        src_proj = torch.matmul(src_in, src_ex[:, 0:3, :])# src_proj [B 3 4]
        ref_proj = torch.matmul(ref_in, ref_ex[:, 0:3, :])
        # 这一步是为了将src_proj补成B 4 4而准备
        last = torch.tensor([[[0,0,0,1.0]]]).repeat(len(src_in),1,1).cuda()# len(src_in)计算batch
        # 将src_proj补成B 4 4
        src_proj = torch.cat((src_proj, last), 1)
        ref_proj = torch.cat((ref_proj, last), 1)

        # 根据两个相机的单应性矩阵计算ref视角图像与src视角图像之前的旋转矩阵和平移矩阵
        proj = torch.matmul(src_proj, torch.inverse(ref_proj)) #这个矩阵理论上是计算ref到src的！！！
        rot = proj[:, :3, :3]  # [B,3,3] 取左上角三行三列得到旋转变换
        trans = proj[:, :3, 3:4]  # [B,3,1] 取最后一列的上面三行得到平移变换

        ##根据图像的像素位置计算旋转后的关系
        # 按照src图像维度构建一张空的平面，之后要做的是根据投影矩阵把src中的像素映射到这张平面上
        # torch.meshgrid输出了两个和src图像同样大小的矩阵，但是其内部元素已被x y初始化
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_feature.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_feature.device)])
        #保证开辟的新空间是连续的(数组存储顺序与按行展开的顺序一致，transpose等操作是跟原tensor共享内存的)，确保后续view操作不会报错
        y, x = y.contiguous(), x.contiguous()
        # 将维度变换为图像样子
        y, x = y.view(height * width), x.view(height * width)
        # torch.ones_like：生成与x相同形状的全1张量，即3个H*W大小的一维张量堆叠在一起
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        #unsqueeze先将维度变为[1, 3, H*W], repeat是为了将batch的维度引入进来
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        # 先将空白空间乘以旋转矩阵
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        # [B, 3, Ndepth, H*W] 再引入Ndepths维度，并将深度假设值填入这个维度 元素点乘
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_hypos.view(batch, 1, num_depth,
                                                                                           1)  # [B, 3, Ndepth, H*W]
        # 旋转变换后的矩阵+平移矩阵 -> 投影变换后的空白平面
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # xy分别除以z进行归一
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        # 换算采样索引
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1 #x方向按照宽度进行归一
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1 #y方向按照高度进行归一
        # 再把归一化后的x和y拼起来
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
        # src_fea(src图像的特征)): [B, C, H, W]
        # 应用双线性插值，把输入的tensor转换为指定大小
        # 按照grid中的映射关系，将src的特征图进行投影变换
        warped_src_fea = F.grid_sample(src_feature, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    # 将上一步编码到height维的深度信息独立出来
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    return warped_src_fea

# 金字塔refine阶段计算更精细的深度假设
#实验中，计算图像中距离0.5像素的点的平均深度间隔
#对于某个像素的深度剩余，首先把它投影到src上，在src的极线上找相邻的两个像素，沿着src相机通过这两个点到3D空间，就得到了深度剩余refine时候的范围
# 训练阶段：精细化的深度假设 = 刚刚得到的粗糙深度图值 ± 4 * intervel
# 测试阶段：思路还是在粗糙的深度图值附近偏移一点 根本不是给人看的 写的太乱太复杂了...
# calDepthHypo(self.args, depth_up, ref_in_multiscales[:, level, :, :],
                # src_in_multiscales[:, :, level, :, :], ref_ex, src_ex, depth_min, depth_max,level)
# depth_up（ref_depths）:上采样后的深度图 [B H W]
# ref_intrinsics:[B,3,3]
# src_intrinsics:[B nsrc 3 3]
# ref_ex:[B 4 4]
# src_ex:[B,nsrc,4,4]
# level:当前的金字塔层
def calDepthHypo(netArgs, ref_depths, ref_intrinsics, src_intrinsics, ref_extrinsics, src_extrinsics,
                 depth_min, depth_max, level):
    ##上述代码主要执行以下操作：
    #计算相机坐标系下的网格点坐标，其中 xx 和 yy 是表示 x 和 y 坐标的张量。
    #根据相机内参和外参，将网格点坐标转换为世界坐标系下的射线方程。
    #将射线方程转换到源图像坐标系。
    #将射线方程转换到源图像像素坐标系，并计算斜率 k 和截距 b。
    #计算步长，并在源图像像素坐标系上计算新的射线方程点 X3。
    #计算参考图像和源图像之间的单应矩阵，并将点 X1 和 X3 正投影回参考图像。
    #构建线性方程组，并求解线性方程组得到 delta_d，即深度偏差。
    #根据深度偏差和统计区间图计算深度假设图。
    ## Calculate depth hypothesis maps for refine steps
    nhypothesis_init = 48
    d = 4
    pixel_interval = 2

    nBatch = ref_depths.shape[0]
    height = ref_depths.shape[1]
    width = ref_depths.shape[2]
    # print('ref_depths: {}'.format(ref_depths.shape))

    with torch.no_grad():
        ref_depths = ref_depths
        ref_intrinsics = ref_intrinsics.double()#转成双精度ref_intrinsics:[B,3,3]
        src_intrinsics = src_intrinsics.squeeze(1).double()#[B nsrc 3 3]→[B 3 3]
        ref_extrinsics = ref_extrinsics.double()#同
        src_extrinsics = src_extrinsics.squeeze(1).double()#同

        interval_maps = []
        depth_hypos = ref_depths.unsqueeze(1).repeat(1, d * 2, 1, 1).double()# [B 8=(2*d) H W]
        for batch in range(nBatch):# 针对batch中的每个深度图进行处理
            # 计算相机坐标系下的xyz坐标值
            # torch.meshgrid（）的功能是生成网格，可以用于生成坐标
            # 其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；第二个输出张量填充第二个输入张量中的元素各列元素相同。
            # >>> xx
            # tensor([[0, 0, 0],
            #         [1, 1, 1]], device='cuda:0')
            # >>> yy
            # tensor([[0, 1, 2],
            #         [0, 1, 2]], device='cuda:0')
            xx, yy = torch.meshgrid([torch.arange(0, width).cuda(), torch.arange(0, height).cuda()])# xx:[W H] yy:[W H]
            # 将xx和yy变成一维张量并且转为双精度 xxx:[W*H]
            xxx = xx.reshape([-1]).double()
            yyy = yy.reshape([-1]).double()
            X = torch.stack([xxx, yyy, torch.ones_like(xxx)], dim=0)#[3 W*H]
            #X：
            #tensor([[0., 0., 0., 1., 1., 1.],
                    #[0., 1., 2., 0., 1., 2.],
                    #[1., 1., 1., 1., 1., 1.]], device='cuda:0', dtype=torch.float64)
            # 根据上面X的Tensor例子，每一列就代表一个像素的坐标 不知道最后的那一行是什么意思，那么X就可以视为连续的(0,0,1)(0,1,1)(0,2,1)...
            # Transpose before reshape to produce identical results to numpy and matlab version.
            D1 = torch.transpose(ref_depths[batch, :, :], 0, 1).reshape([-1]).double()  # [B H W]→[H W]→[W H]→[H*W] 此时D1中保存的是粗略深度值
            D2 = D1 + 1 # D1中的元素逐个加1，这个D2可能是为后面求k服务的，因为满足某种三角关系

            # 乘上深度（z坐标，对齐尺度）
            X1 = X * D1 # [3 W*H]*[H*W] 元素点乘 可以视为z1*(0,0,1);z2*(0,1,1);....即Z*(x,y,1)，这一步应该是把深度图变成3D点。
            X2 = X * D2 # 同上

            # 乘上ref相机内参，得到相机坐标系下的xyz坐标(# 计算相机坐标系下的射线方程)
            ray1 = torch.matmul(torch.inverse(ref_intrinsics[batch]), X1) # ref_intrinsics[batch]:[3 3] ray1：[3 H*W]
            ray2 = torch.matmul(torch.inverse(ref_intrinsics[batch]), X2) # ray2：[3 H*W]

            # 乘上ref相机外参，校正到世界坐标系
            X1 = torch.cat([ray1, torch.ones_like(xxx).unsqueeze(0).double()], dim=0) #X1:[3 H*W]→[4 H*W],为ray1添加一行1在最后
            X1 = torch.matmul(torch.inverse(ref_extrinsics[batch]), X1)# [4 H*W]
            X2 = torch.cat([ray2, torch.ones_like(xxx).unsqueeze(0).double()], dim=0)
            X2 = torch.matmul(torch.inverse(ref_extrinsics[batch]), X2)# [4 H*W]

            # 乘上src相机外参，转到src相机坐标系
            X1 = torch.matmul(src_extrinsics[batch][0], X1) #[4 4]×[4 H*W]=[4 H*W]
            X2 = torch.matmul(src_extrinsics[batch][0], X2)

            # 乘上src相机内参，转到src图像坐标系
            X1 = X1[:3] #[3 H*W] 舍去了最后一行添加的1
            X1 = torch.matmul(src_intrinsics[batch][0], X1)
            # 除以z坐标（深度）转换到图像坐标系 去除深度这个信息
            X1_d = X1[2].clone()
            X1 /= X1_d

            X2 = X2[:3]
            X2 = torch.matmul(src_intrinsics[batch][0], X2)
            X2_d = X2[2].clone()
            X2 /= X2_d

            # 上面的步骤将在ref图像坐标系上的深度图转换到了src图像坐标系 即反投影
            # 现在X1和X2只有前两个行有用，最后一行都是1
            # 计算斜率和截距
            k = (X2[1] - X1[1]) / (X2[0] - X1[0])#k:[H*W] 其中每个元素表示源图像上相应直线的斜率。
            b = X1[1] - k * X1[0] #b同理
            #下面的计算中X1已经不含有深度信息了
            theta = torch.atan(k)
            # 计算步长。X3 的作用是通过沿着射线方向添加一个固定的像素间隔来计算一个新的点。
            X3 = X1 + torch.stack(
                [torch.cos(theta) * pixel_interval, torch.sin(theta) * pixel_interval, torch.zeros_like(X1[2, :])],
                dim=0) #[3 H*W]

            # 计算单应矩阵
            A = torch.matmul(ref_intrinsics[batch], ref_extrinsics[batch][:3, :3])# 参考图像单应矩阵
            tmp = torch.matmul(src_intrinsics[batch][0], src_extrinsics[batch][0, :3, :3]) #源图像单应矩阵
            A = torch.matmul(A, torch.inverse(tmp))#源图像投影到参考图像的转换 正投影

            #正投影
            tmp1 = X1_d * torch.matmul(A, X1)# 将X1再正投影回去 不含有深度信息
            tmp2 = torch.matmul(A, X3) #将X3正投影回去 A：[3 3] X3:[3 H*W]
            # torch.t()是一个类似于求矩阵的转置的函数，但是它要求输入的tensor结构维度 <= 2D。

            # 构建线性方程组
            # 首先，根据参考图像和源图像之间的投影关系，构建了一个线性方程组。
            # 线性方程组的形式为 M1 * delta_d = M2。
            # 其中 M1 是一个大小为 [H*W, 2, 2] 的矩阵，M2 是一个大小为 [H*W, 2] 的向量，delta_d 是一个长度为 H*W 的待求解的向量。
            # 具体来说，通过反投影操作，将参考图像中的某个像素点 X1 和其对应的新点 X3 投影到源图像上。
            # 然后，使用参考图像和源图像的相机内参、外参，以及射线方程的关系，将这些点转换到图像坐标系上。
            # 在这个过程中，通过计算 X3 和 X1 在图像坐标系上的坐标值之间的关系，得到了 M1 矩阵。
            # 同时，将 X1 投影回参考图像的过程得到了 M2 向量。
            # 最后，通过求解线性方程组 M1 * delta_d = M2，即可得到 delta_d，它表示源图像中每个像素点相对于参考图像的深度偏移量。
            M1 = torch.cat([X.t().unsqueeze(2), tmp2.t().unsqueeze(2)], dim=2)[:, 1:, :]#X.t().unsqueeze(2)：[H*W 3 1] M1：[H*W 2 2]
            M2 = tmp1.t()[:, 1:]# (H*W 2)
            # print('M1: {} M2: {}'.format(M1.shape, M2.shape))
            # print('torch.inverse(M1): {}'.format(torch.inverse(M1).shape))
            # print('M2.unsqueeze(2): {}'.format(M2.unsqueeze(2).shape))
            tmp1 = torch.inverse(M1)#[H*W 2 2]
            tmp2 = M2.unsqueeze(2)#[H*W 2 1]
            ans = torch.matmul(tmp1, tmp2)#[H*W 2 1]
            # ans = torch.bmm(tmp1, tmp2)
            delta_d = ans[:, 0, 0] #delta_d：[H*W]

            #torch.abs(delta_d)：对 delta_d 进行取绝对值，得到绝对值的张量。
            #.mean()：计算绝对值张量的平均值，得到一个标量。
            #.repeat(ref_depths.shape[2], ref_depths.shape[1])：将标量值重复复制为指定的维度，重复次数由 ref_depths 张量的维度决定。这将得到一个与 ref_depths 张量具有相同形状的张量，其中每个元素都是平均值。
            #.t()：对重复后的张量进行转置操作，将维度进行交换。这将使得结果张量的维度为 [W, H]，与 ref_depths 张量的形状相匹配。
            interval_maps = torch.abs(delta_d).mean().repeat(ref_depths.shape[2], ref_depths.shape[1]).t()

            for depth_level in range(-d, d):
                depth_hypos[batch, depth_level + d, :, :] += depth_level * interval_maps

            # print("Calculated:")
            # print(interval_maps[0,0])

            # pdb.set_trace()

        return depth_hypos.float()  # Return the depth hypothesis map from statistical interval setting.

# 金字塔refine阶段计算cost volume剩余
# 跟homo_warping步骤基本一模一样，区别在于聚集每个src最高分辨率特征体warp后的volume
def proj_cost(settings, ref_feature, src_feature, level, ref_in, src_in, ref_ex, src_ex, depth_hypos):
    ## Calculate the cost volume for refined depth hypothesis selection
    batch, channels = ref_feature.shape[0], ref_feature.shape[1]
    num_depth = depth_hypos.shape[1]
    height, width = ref_feature.shape[2], ref_feature.shape[3]
    nSrc = len(src_feature)

    volume_sum = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
    volume_sq_sum = volume_sum.pow_(2)

    for src in range(settings.nsrc):
        with torch.no_grad():
            src_proj = torch.matmul(src_in[:, src, :, :], src_ex[:, src, 0:3, :])
            ref_proj = torch.matmul(ref_in, ref_ex[:, 0:3, :])
            last = torch.tensor([[[0, 0, 0, 1.0]]]).repeat(len(src_in), 1, 1).cuda()
            src_proj = torch.cat((src_proj, last), 1)
            ref_proj = torch.cat((ref_proj, last), 1)

            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3]
            trans = proj[:, :3, 3:4]

            y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=ref_feature.device),
                                   torch.arange(0, width, dtype=torch.float32, device=ref_feature.device)])
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(height * width), x.view(height * width)
            xyz = torch.stack((x, y, torch.ones_like(x)))
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)
            rot_xyz = torch.matmul(rot, xyz)

            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_hypos.view(batch, 1, num_depth,
                                                                                               height * width)  # [B, 3, Ndepth, H*W]
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)
            proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
            proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
            proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
            proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)
            grid = proj_xy
        warped_src_fea = F.grid_sample(src_feature[src][level], grid.view(batch, num_depth * height, width, 2),
                                       mode='bilinear',
                                       padding_mode='zeros')
        warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

        volume_sum = volume_sum + warped_src_fea
        volume_sq_sum = volume_sq_sum + warped_src_fea.pow_(2)
    cost_volume = volume_sq_sum.div_(settings.nsrc + 1).sub_(volume_sum.div_(settings.nsrc + 1).pow_(2))

    if settings.mode == "test":
        del volume_sum
        del volume_sq_sum
        torch.cuda.empty_cache()

    return cost_volume


# MVSNet modules
# 二维卷积涉及模块
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=True),
        # LeakyReLU与ReLU很相似，仅在输入小于0的部分有差别，ReLU输入小于0的部分值都为0，
        # LeakyReLU输入小于0的部分，值为负，且有微小的梯度。
        # 实际中，LeakyReLU的α取值一般为0.01。
        nn.LeakyReLU(0.1)
    )
# 定义注意力模块
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)  # 1x1卷积用于特征压缩
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)  # 使用1x1卷积压缩特征
        attention = self.sigmoid(attention)  # 使用sigmoid激活函数
        out = x * attention  # 特征加权
        return out

# 三维卷积涉及模块
class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=True)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

#根据上面模块加入自注意力机制
class SelfAttentionConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(SelfAttentionConvBnReLU3D, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=True)
        self.attention = nn.Sequential(
            nn.Conv3d(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels // 2, 1, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=2)  # 在特征图维度上进行softmax
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x_conv = self.conv(x)

        # 使用注意力层在特征图维度上生成权重
        attn_weights = self.attention(x_conv)

        # 将注意力权重应用于特征图
        x_attn = x_conv * attn_weights

        # 应用批归一化和ReLU
        x_attn = self.bn(x_attn)
        x_attn = F.relu(x_attn, inplace=True)
        return x_attn


# 深度回归涉及模块
def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth

def depth_regression_refine(prob_volume, depth_hypothesis):
    return torch.sum(prob_volume * depth_hypothesis, 1)






















##没有用到以下模块
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out

class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=True)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

