from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import GCN
import numpy as np
import utils.util as util


class AttModel(Module):
    # 这里的dctn设置为了10 但是项目提供的预训练模型却是20
    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10

        # Sequential 就是容器 可以容纳多个网络层
        # self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
        #                                      bias=False),
        #                            nn.ReLU(),
        #                            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
        #                                      bias=False),
        #                            nn.ReLU())
        #
        # self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
        #                                      bias=False),
        #                            nn.ReLU(),
        #                            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
        #                                      bias=False),
        #                            nn.ReLU())
        # 自己写的gcn
        #                                  咋*2了
        # 因为 最后的"最新子序" 与 "dct_att_tmp" 的cancat操作是在最后一维上的 也就是 dctn + dctn
        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        # linear
        # 修改点1: Q线性层 - 输入维度为kernel_size，在序列长度维度上做线性变换
        self.q_linear = nn.Linear(kernel_size, 1, bias=True)

        # 修改点2: K线性层使用动态初始化，因为其输入维度取决于input_n-output_n
        # 在forward中根据实际的输入维度动态创建
        self.k_linear = nn.Linear(15, 1, bias=True)

    def forward(self, src,
                # 代码结构怪异 在跑eval时得把dct_n解除注释 以接收
                # 但是跑train时 却是没有输入dctn这个参数
                dct_n,
                output_n=25, input_n=50, itera=1):
        """

        :param src: [batch_size,seq_len,feat_dim]  # source 需要输入一个batch数据包
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:   迭代次数 区别于 epoch
        :return:
        """
        dct_n = self.dct_n
        # 这里是语法省略 第二维取到input 然后第3维全取 所以后面的第3维的切片省略不写
        src = src[:, :input_n]  # [bs,in_n,dim]
        # 暂时复制
        src_tmp = src.clone()
        # 获取bat的大小
        bs = src.shape[0]
        # 将 2 3 维进行转置
        # :n 是前面n个   -n: 是倒数n个
        # 为什么这里的k是整个大序列的前input_n - output_n个？？
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        # 论文里定义最新的子序列（最后一个窗口）的倒数M（k-size）个作为Q
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        # 获取dct矩阵
        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        # 对接cuda
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        # 滑动窗口的数量(这里先按照 input是一个样本的所有帧 kernelsize是前M个历史帧 output是后T帧 来理解)
        vn = input_n - self.kernel_size - output_n + 1
        # 一个窗口下 一个序列的长度
        vl = self.kernel_size + output_n
        # 构建滑动窗口的索引
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        # 假设 vn=3, vl=5
        # np.arange(vl) = [0,1,2,3,4]
        # np.arange(vn) = [0,1,2]
        # idx = [[0,1,2,3,4],   # 窗口1：起始位置0
        #        [1,2,3,4,5],   # 窗口2：起始位置1
        #        [2,3,4,5,6]]   # 窗口3：起始位置2

        # 按滑动窗口提取 然后进行维度变化 将vn维与batch归并
        # 每个窗口下的整个sub-sqe就是v
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        # 进行dct
        # 由于是乘法操作 无法像加法那样进行广播 所以dct需要进行升维
        # 取前面dct_n行
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,31,48*11] 我觉得是10 不是 11可能打错了
        # 比如：最后转为 32个bat 31个滑动窗口 48个位置信息*10个dct系数 一个窗口下的运动信息 变成10个dct系数

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        # eg    后面的output帧 对ks最后的索引进行了复制
        #   [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
        #    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #    -1, -1, -1, -1, -1]

        outputs = []

        # 这里做卷积的是没有做dct的src_key_tmp 和 src_query_tmp
        # 奇怪 Q K 竟是用卷积出来的 不是用linear 难道等价？

        # 修改点3: 动态创建或检查K线性层

        if self.k_linear is None or self.k_linear.in_features != input_n - output_n:
            # 动态创建K线性层，输入维度为seq_len_k
            self.k_linear = nn.Linear(input_n - output_n, vn, bias=True).cuda()

        # 修改点4: 调整输入维度以匹配线性层
        # src_key_tmp形状: [bs, in_features, seq_len_k]
        # 需要转置为: [bs, in_features, seq_len_k] -> [bs, in_features, seq_len_k] 保持原状
        # 但线性层期望的输入形状为: [*, seq_len_k]，所以需要交换维度
        key_tmp = self.k_linear(src_key_tmp)


        for i in range(itera):  # 迭代循环
            # 修改点5: 调整Q线性层的输入
            # src_query_tmp形状: [bs, in_features, kernel_size]
            query_tmp = self.q_linear(src_query_tmp)
            query_tmp = query_tmp.transpose(1, 2)

            # 计算得分
            # 修改点6: 注意现在query_tmp和key_tmp的形状都是[bs, 1, in_features]
            # 所以需要调整矩阵乘法的维度
            score_tmp = torch.matmul(query_tmp, key_tmp) + 1e-15
            # 归一
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            # 计算att*v
            # 这里的v有做dct 获取q k的seq 没做dct
            # 跨域处理？？？               得分     每个subsqe的v
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])

            # 取最后的序列（含最后一帧复制）
            input_gcn = src_tmp[:, idx]
            # 做dct
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
            # concat
            # concat后 上部10个dct是最后一个带复制帧的子序 下部10个是att后的输出
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
            # 走GCN图卷积
            dct_out_tmp = self.gcn(dct_in_tmp)
            # 进行idct返回
            # 因为前面进行了conact操作 从10-20  这里在进行取前面10个
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),  # 在位置0 添加一个维度 也就是bat维
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            # 在位置2 添加一个维度原本 原本 "bat 帧 位置信息"  -> "bat itera 帧 位置信息"
            # 添加的维度用来后面进行 itera的记录
            outputs.append(out_gcn.unsqueeze(2))

            # 假如有迭代 拼接预测帧 然后更新vn vl 最后更新qkv的对象
            if itera > 1:
                # update key-value query

                # 取预测出的子序的倒数output个 也就是t
                out_tmp = out_gcn.clone()[:, 0 - output_n:]
                # 将预测的拼接上
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)

                vn = 1 - 2 * self.kernel_size - output_n
                vl = self.kernel_size + output_n
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                          np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)

                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)
                # 修改点7: 更新K时同样需要调整维度
                key_new = self.k_linear(src_key_tmp.transpose(1, 2))
                key_new = key_new.transpose(1, 2)
                key_tmp = torch.cat([key_tmp, key_new], dim=2)

                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)

                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)

        # 在之前创建的维度2 进行cancat
        outputs = torch.cat(outputs, dim=2)
        return outputs