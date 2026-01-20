from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import GCN
import numpy as np
import utils.util as util


class AttModel(Module):
    # 默认参数 若有改动 则覆盖
    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=20):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10

        self.linear_k = nn.Linear(48*5, d_model)
        self.linear_k1 = nn.Linear(48 * 5, d_model)
        self.linear_k2 = nn.Linear(48 * 5, d_model)
        self.linear_k3 = nn.Linear(48 * 5, d_model)

        self.linear_q = nn.Linear(48*5, d_model)
        self.linear_q1 = nn.Linear(48*5, d_model)
        self.linear_q2 = nn.Linear(48*5, d_model)
        self.linear_q3 = nn.Linear(48*5, d_model)

        self.linear_w0 = nn.Linear(80, 20)

        #                                  咋*2了
        # 因为 最后的“最新子序” 与 “dct_att_tmp” 的cancat操作是在第2维上的 也就是 dctn + dctn
        self.gcn = GCN.GCN(input_feature=dct_n * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

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
        src_tmp = src.clone()  # [bs,in_n,dim]
        # 获取bat的大小
        bs = src.shape[0]
        # 获取dct矩阵
        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_kq, idct_kq = util.get_dct_matrix(self.kernel_size) # ks = output
        # 对接cuda
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()
        dct_kq = torch.from_numpy(dct_kq).float().cuda()
        idct_kq = torch.from_numpy(idct_kq).float().cuda()
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

        # 论文里定义最新的子序列（最后一个窗口）的倒数M（k-size）个作为Q
        # print("q.shape==================")
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()
        # print(src_query_tmp.shape)
        src_query_tmp = src_query_tmp.transpose(1, 2)
        # print(src_query_tmp.shape)
        src_query_tmp = torch.matmul(dct_kq[:5].unsqueeze(dim=0), src_query_tmp).transpose(1, 2)
        # print(src_query_tmp.shape)
        src_query_tmp = src_query_tmp.reshape(bs,1,-1)
        # print(src_query_tmp.shape)


        # 按滑动窗口提取 然后进行维度变化 将vn维与batch归并
        # 每个窗口下的整个sub-sqe就是v
        # print("k.shape==================")
        src_key_tmp   = src_tmp[:,idx].clone().reshape(bs,vn,vl,-1)
        # print(src_key_tmp.shape)
        src_key_tmp = src_key_tmp[:,:,0:10,:]
        # print(src_key_tmp.shape)
        src_key_tmp = src_key_tmp.reshape(bs*vn,10,48)
        # print(src_key_tmp.shape)
        src_key_tmp = torch.matmul(dct_kq[:5].unsqueeze(dim=0), src_key_tmp)
        # print(src_key_tmp.shape) # 这里由于ks=10 所以dct系数只能设置<=10 数学约束
        src_key_tmp = src_key_tmp.reshape(bs,vn,-1)
        # print(src_key_tmp.shape)

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

        key_tmp = self.linear_k(src_key_tmp)
        key_tmp1 =self.linear_k1(src_key_tmp)
        key_tmp2 = self.linear_k2(src_key_tmp)
        key_tmp3 = self.linear_k3(src_key_tmp)
        for i in range(itera): # 迭代循环
            query_tmp = self.linear_q(src_query_tmp)
            query_tmp1 = self.linear_q1(src_query_tmp)
            query_tmp2 = self.linear_q2(src_query_tmp)
            query_tmp3 = self.linear_q3(src_query_tmp)



            # 计算得分
            score_tmp = torch.matmul(query_tmp, key_tmp.transpose(1,2)) + 1e-15
            score_tmp1 = torch.matmul(query_tmp1, key_tmp1.transpose(1, 2)) + 1e-15
            score_tmp2 = torch.matmul(query_tmp2, key_tmp2.transpose(1, 2)) + 1e-15
            score_tmp3 = torch.matmul(query_tmp3, key_tmp3.transpose(1, 2)) + 1e-15
            # 归一
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            att_tmp1 = score_tmp1 / (torch.sum(score_tmp1, dim=2, keepdim=True))
            att_tmp2 = score_tmp2 / (torch.sum(score_tmp2, dim=2, keepdim=True))
            att_tmp3 = score_tmp3 / (torch.sum(score_tmp3, dim=2, keepdim=True))



            # 计算att*v
            # 这里的v有做dct 获取q k的seq 没做dct
            #                            得分     每个subsqe的v
            # 谜之操作 我觉得取第2维的第一个 可以省去 毕竟第2维只有1行啊
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])
            dct_att_tmp1 = torch.matmul(att_tmp1, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])
            dct_att_tmp2 = torch.matmul(att_tmp2, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])
            dct_att_tmp3 = torch.matmul(att_tmp3, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])
            dct_att_tmp = torch.cat([dct_att_tmp,dct_att_tmp1, dct_att_tmp2, dct_att_tmp3], dim=-1)
            dct_att_tmp = self.linear_w0(dct_att_tmp)

            # 取最后的序列（含最后一帧复制）
            input_gcn = src_tmp[:, idx]
            # 做dct
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
            # concat
            # concat后 左10个dct是最后一个带复制帧的子序 右10个是att后的输出
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
            # 走GCN图卷积
            dct_out_tmp = self.gcn(dct_in_tmp)
            # 进行idct返回
            # 因为前面进行了conact操作 从10-20  这里在进行取前面10个
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),  # 为 idct矩阵 在位置0 添加一个维度 也就是bat维
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            # 最后输出(bat 帧 位置信息)

            # 在位置2 添加一个维度原本 原本 “bat 帧 位置信息”  -》 “bat itera 帧 位置信息”
            # 添加的维度用来后面进行 itera的记录
            outputs.append(out_gcn.unsqueeze(2))

            # 假如有迭代 拼接预测帧 然后更新vn vl 最后更新qkv的对象
            # 训练时没迭代 验证时有
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
                key_new = self.convK(src_key_tmp / 1000.0)
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

