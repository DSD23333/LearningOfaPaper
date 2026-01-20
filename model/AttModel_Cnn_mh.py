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
        self.heads = 4
        assert kernel_size == 10

        # 卷积list
        self.convQ_list = nn.ModuleList()
        self.convK_list = nn.ModuleList()
        # 填充list
        for _ in range(self.heads):
            # 每个头的Q卷积
            convQ = nn.Sequential(
                nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6, bias=False),
                nn.ReLU(),
                nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False),
                nn.ReLU()
            )

            # 每个头的K卷积
            convK = nn.Sequential(
                nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6, bias=False),
                nn.ReLU(),
                nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False),
                nn.ReLU()
            )
            self.convQ_list.append(convQ)
            self.convK_list.append(convK)

        #                                  咋*2了
        # 因为 最后的“最新子序” 与 “dct_att_tmp” 的cancat操作是在最后一维上的 也就是 dctn + dctn
        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.linear = nn.Linear(in_features=48*4, out_features= 48 )

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
        #比如：最后转为 32个bat 31个滑动窗口 48个位置信息*10个dct系数 一个窗口下的运动信息 变成10个dct系数

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        # eg    后面的output帧 对ks最后的索引进行了复制
        #   [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
        #    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #    -1, -1, -1, -1, -1]

        outputs = []

        # 这里做卷积的是没有做dct的src_key_tmp 和 src_query_tmp

        # 尝试实现生成多个k=========================================
        key_list = []
        for convK in self.convK_list:
            key_tmp_head = convK(src_key_tmp / 1000.0)
            key_list.append(key_tmp_head)

        #============================================================

        for i in range(itera): # 迭代循环

            # 多头============================================
            query_list = []
            for convQ in self.convQ_list:
                query_tmp_head = convQ(src_query_tmp / 1000.0)
                query_list.append(query_tmp_head)
            #==========================================================

            att = []

            # 计算得分
            for i in range(self.heads):
                score_tmp = torch.matmul(query_list[i].transpose(1, 2), key_list[i]) + 1e-15
                # 归一
                att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
                # 计算att*v
                # 这里的v有做dct 获取q k的seq 没做dct
                # 跨域处理？？？               得分     每个subsqe的v
                dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape([bs, -1, dct_n])
                att.append(dct_att_tmp)
            att = torch.cat(att, dim=1)
            att = self.linear(att.transpose(1, 2)).transpose(1, 2)

            # 取最后的序列（含最后一帧复制）
            input_gcn = src_tmp[:, idx]
            # 做dct
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
            # concat
            # concat后 左部10个dct是最后一个带复制帧的子序 右部10个是att后的输出
            dct_in_tmp = torch.cat([dct_in_tmp, att], dim=-1)
            # 走GCN图卷积
            dct_out_tmp = self.gcn(dct_in_tmp)
            # 进行idct返回
            # 因为前面进行了conact操作 从10-20  这里在进行取前面10个
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),  # 在位置0 添加一个维度 也就是bat维
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            # 在位置2 添加一个维度原本 原本 “bat 帧 位置信息”  -》 “bat itera 帧 位置信息”
            # 添加的维度用来后面进行 itera的记录
            outputs.append(out_gcn.unsqueeze(2))

            # 假如有迭代 拼接预测帧 然后更新vn vl 最后更新qkv的对象
            # 目前在eval里迭代是3 但是 最后文csv文件的输出却只输出一个迭代
            # 所以改模型后假若只要看第一迭代的eval数据 后面这部分可以不改
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
                # key_new = self.convK(src_key_tmp / 1000.0)
                # key_tmp = torch.cat([key_tmp, key_new], dim=2)

                # 这里也是要按照多头的更新办法
                key_new_list = []
                for convK in self.convK_list:
                    key_new_head = convK(src_key_tmp / 1000.0)
                    key_new_list.append(key_new_head)

                # 拼接多头K
                key_new_multi = torch.cat(key_new_list, dim=1)

                # 转置并应用融合线性层
                key_new_multi_transposed = key_new_multi.transpose(1, 2)
                key_new_fused = self.k_fusion_linear(key_new_multi_transposed)
                key_new_final = key_new_fused.transpose(1, 2)

                # 拼接新的K到已有的K中
                key_tmp_final = torch.cat([key_tmp_final, key_new_final], dim=2)




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
