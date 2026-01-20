from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils
from matplotlib import pyplot as plt
import torch


class Datasets(Dataset):

    def __init__(self, opt, actions=None, split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation  目标任务
        :param sample_rate:  下采样
        """
        # 这里暂时使用了绝对路径
        self.path_to_data = "D:\code\yolo\HisRepItself-read-oper\datasets\h3.6m"
        self.split = split   # 任务索引
        self.in_n = opt.input_n
        self.out_n = opt.output_n
        self.sample_rate = 2     # 这里并没有开放可在之后自定义采样率的接口 直接订死了
        # 先创建空的
        self.seq = {}   # 字典
        self.data_idx = []
        # 这里定义考虑计算的关节
        # 相比与paper0的计算后再决定 这里直接无脑定义数组 省时省力
        self.dimensions_to_use = np.array(
            [6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
             43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85, 86])
        self.dimensions_to_ignore = np.array(
            [[0, 1, 2, 3, 4, 5,
              10, 11,
              16, 17, 18, 19, 20,
              25, 26,
              31, 32, 33, 34, 35,
              48, 49, 50,
              58,59,
              63, 64, 65, 66, 67, 68, 69,
              70, 71, 72, 73, 74,
              82, 83, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
              98]])

        # 帧序列的长度 对应 M + T
        seq_len = self.in_n + self.out_n
        # # 对数据集进行分类
        # subs = np.array([[1, 6, 7, 8, 9],  # 训练
        #                  [11],             # 这个是每个epoch后的验证
        #                  [5]])             # 最终与其他模型的对比test
        # 这里进行了语法修改
        subs = [[1, 6, 7, 8, 9],  # 训练
                [11],  # 这个是每个epoch后的验证
                [5]]# 最终与其他模型的对比test


        # acts = data_utils.define_actions(actions)
        if actions is None:
            # 空参数则设定为全部
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
            # 有指定就指定
        else:
            acts = actions
        # subs = np.array([[1], [11], [5]])
        # acts = ['walking']

        # 简陋的任务区分 前面 将数据集进行分类 然后split就是检索。。。
        subs = subs[split]

        for subj in subs: # 对于演员
            for action_idx in np.arange(len(acts)):  # 对于动作
                action = acts[action_idx]
                if self.split <= 1:  # 对于 训练 验证 数据集
                    for subact in [1, 2]:  # subactions 在这里每个演员的每个动作都有 2 个版本

                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)

                        # readCSVasFloat 的参数就直接传 文件路径 ！
                        # eg：the_sequence == [  [1,2,6] , [5,9,3] , [..]... ]
                        # 之后 the_sequence 第一维是帧 第二维是“每行”关节数据
                        the_sequence = data_utils.readCSVasFloat(filename)

                        # n=帧数  d==99
                        n, d = the_sequence.shape
                        # 为了下采样的 index 标记创建
                        even_list = range(0, n, self.sample_rate)
                        # 记录下采样后的帧数
                        num_frames = len(even_list)

                        # 取下采样指定的index
                        # 第一维之只取 index 第二维(d==99) 全取
                        # the_sequence 更新
                        the_sequence = np.array(the_sequence[even_list, :])

                        # the_sequence = torch.from_numpy(the_sequence).float().cuda()
                        # 这里似乎是从numpy到cuda的兼容？
                        # remove global rotation and translation
                        # 移除全局移动和旋转 直接设置为0？为什么不直接舍去呢
                        the_sequence[:, 0:6] = 0
                        # p3d = data_utils.expmap2xyz_torch(the_sequence) 对于p3d数据集

                        # 给之前创建的字典进行app 以及赋值
                        self.seq[(subj, action, subact)] = the_sequence

                        # 这里的步长是对全帧进行下采样后的 进一步对所有的样本进行“跳过”
                        # 似乎是从一个相对连续的大动作序列 进行分割 分出相对独立的小序列
                        #                        起始     终止                   步长
                        valid_frames = np.arange(0, num_frames - seq_len + 1, opt.skip_rate)

                        tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                        # len(valid_frames个相同的(subj, action, subact)
                        tmp_data_idx_2 = list(valid_frames)
                        # zip 匹配
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        # eg
                        # zipped = [
                        #     (('S1', 'walking', '01'), 0),
                        #     (('S1', 'walking', '01'), 10),
                        #     (('S1', 'walking', '01'), 20),
                        #     (('S1', 'walking', '01'), 30),
                        #     (('S1', 'walking', '01'), 40),
                        #     (('S1', 'walking', '01'), 50),
                        #     (('S1', 'walking', '01'), 60),
                        #     (('S1', 'walking', '01'), 70),
                        #     (('S1', 'walking', '01'), 80)
                        # ]


                else:
                    #================sub1======================
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 1)
                    the_sequence1 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1[even_list, :])
                    # the_seq1 = torch.from_numpy(the_sequence1).float().cuda()
                    the_sequence1[:, 0:6] = 0
                    # p3d1 = data_utils.expmap2xyz_torch(the_seq1)
                    self.seq[(subj, action, 1)] = the_sequence1
                    #===================sub2====================
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 2)
                    the_sequence2 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2[even_list, :])
                    # the_seq2 = torch.from_numpy(the_sequence2).float().cuda()
                    the_sequence2[:, 0:6] = 0
                    # p3d2 = data_utils.expmap2xyz_torch(the_seq2)
                    self.seq[(subj, action, 2)] = the_sequence2
                    # ==========================================


                    #==========找到与rssn一样的实验片段============

                    # fs_sel1, fs_sel2 = data_utils.find_indices_256(num_frames1, num_frames2, seq_len,
                    #                                                 input_n=self.in_n)
                    fs_sel1, fs_sel2 = data_utils.find_indices_srnn(num_frames1, num_frames2, seq_len,
                                                                    input_n=self.in_n)

                    valid_frames = fs_sel1[:, 0]
                    tmp_data_idx_1 = [(subj, action, 1)] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                    valid_frames = fs_sel2[:, 0]
                    tmp_data_idx_1 = [(subj, action, 2)] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

# 到此为止 我们获得了2个东西: 一个庞大的字典{(subj, action, sub)] = the_sequence....}
#                         每个(subj, action, sub)只出现一次 the_sequence也是下采样 设1-6为0 后完整的
# 注意！！ 就是99个完整的 （只不过1-6 = 0）
# 还有就是 如此一个元组列表   2号元素对应skip后的起始帧
# eg
# zipped = [
#     (('S1', 'walking', '01'), 0),
#     (('S1', 'walking', '01'), 10),
#     (('S1', 'walking', '01'), 20),
#     (('S1', 'walking', '01'), 30),
#     (('S1', 'walking', '01'), 40),
#     (('S1', 'walking', '01'), 50),
#     (('S1', 'walking', '01'), 60),
#     (('S1', 'walking', '01'), 70),
#     (('S1', 'walking', '01'), 80)
# ]

#=========================================================
# 以下就是最后的数据集对 torch的对接
# len 告诉 torch 一共有几个数据样本  --》后续batch处理
# getitem则是 说明一个样本是什么样的 到哪里去拿

    # 返回如上zipped的第1维的长度 也就是样本总量
    def __len__(self):
        # return np.shape(self.data_idx)[0]
        # 语法修改
        return len(self.data_idx)

    def __getitem__(self, item):
        # 对于每个item 也就是元组列表里的元组
        # 进行赋值 key是 哪一个 完整序列
        # start_frame是 起始帧
        key, start_frame = self.data_idx[item]
        #               从起始帧      取 in_put + output的帧长度
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        # 到字典里去找 value 也就是序列
        return self.seq[key][fs]
