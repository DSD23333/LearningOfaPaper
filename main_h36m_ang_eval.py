from utils import h36motion as datasets

from model import AttModel_linear_v2_qk3DctN


from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import h5py
import torch.optim as optim


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = 48
    d_model = opt.d_model
    kernel_size = opt.kernel_size


    net_pred = AttModel_linear_v2_qk3DctN.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)



    # net_pred = AttModel_remake.AttModel_remake(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
    #                              num_stage=opt.num_stage, dct_n=opt.dct_n)




    net_pred.cuda()
    model_path_len = '{}/ckpt_best.pth.tar'.format(opt.ckpt)    # 取best
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    start_epoch = ckpt['epoch'] + 1
    err_best = ckpt['err']
    lr_now = ckpt['lr']
    # 导入模型的参数 state_dict是一个字典 里面包含每个网络里面参数
    net_pred.load_state_dict(ckpt['state_dict'])
    # net.load_state_dict(ckpt)
    # optimizer.load_state_dict(ckpt['optimizer'])
    # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    # 创建表头
    head = np.array(['act'])
    for k in range(1, opt.output_n + 1):
        head = np.append(head, [f'#{k}'])

    acts = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]
    # 全零矩阵
    errs = np.zeros([len(acts) + 1, opt.output_n])
    # 对于数据集下的一个动作
    for i, act in enumerate(acts):
        #                                       eval集   指定动作
        # 一个动作下的序列有进行skip采样 所以最后test_dataset最后是多个样本
        test_dataset = datasets.Datasets(opt, split=2, actions=[act])
        # 所谓 length其实就是在h36motion里最后--len--设置的返回 样本长度
        print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
        test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                 pin_memory=True)
        # 输入到模型运作器 返回数据 打印error 输出的是一个误差输出 #1
        # output的每一帧的误差 先只打印以一个output帧
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        print('testing error: {:.3f}'.format(ret_test['#1']))

        # 创建结果记录
        ret_log = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
        errs[i] = ret_log
    errs[-1] = np.mean(errs[:-1], axis=0) # errs里创建的空位赋值为误差均值

    # 将原本的acts 添加一个 aver表头 意在创建一个表
    acts = np.expand_dims(np.array(acts + ["average"]), axis=1)
    # 这里进行了语法修改 将np.str 改为 str
    # 在表头下 写数值
    value = np.concatenate([acts, errs.astype(str)], axis=1)
    # 在模型文件夹下生成csv记录 head是前面创建的表头
    # 这里实际上是8个测试样本 原来是256 256是作者后面的测试 但是其他的代码却是匹配8的
    # 只是在文件的命名上 作者没改
    log.save_csv_log(opt, head, value, is_create=True, file_name='test_pre_action_8_seq')


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    net_pred.eval()

    titles = np.array(range(opt.output_n)) + 1
    # 创建0数组
    m_ang_seq = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                         43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                         86])
    seq_in = opt.kernel_size

    # 迭代也给订死了
    itera = 1
    idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
            out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    for i, (ang_h36) in enumerate(data_loader):
        # 对于每个batch
        # print(i)
        batch_size, seq_n, _ = ang_h36.shape
        # when only one sample in this batch
        # 如果最后的一个batch只分配到一个样本 就舍去？
        if batch_size == 1 and is_train == 0:
            continue
        # n 不断加bat
        n += batch_size
        bt = time.time()
        # 转f
        # ang_h36 未做dimuse
        ang_h36 = ang_h36.float().cuda()
        # 取最新的子序
        ang_sup = ang_h36.clone()[:, :, dim_used][:, -(out_n + seq_in):]
        # 取一个batch样本 做dimuse
        ang_src = ang_h36.clone()[:, :, dim_used]
        # ang_src = ang_src.permute(1, 0, 2)  # seq * n * dim
        # ang_src = ang_src[:in_n]

        # 走预测网络 输出预测结果（m+t）
        ang_out_all = net_pred(ang_src, output_n=10, dct_n=opt.dct_n,
                               itera=itera, input_n=in_n)
        # 取seq_in
        # 补充一下语法：一个切片操作由3部分组成(start:end:step)，其中step可选
        # 然后假若对象是一个3维的张量 那么就要对应3次切片操作  最后一维不操作可省略
        # ？                          从seq_in的结尾开始选                          10个out 3次迭代(共30)  取第一次迭代
        ang_out_all = ang_out_all[:, seq_in:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:, :out_n]
        # 对ang_h36(未dimuse)取 out 的那一块
        ang_out = ang_h36.clone()[:, in_n:in_n + out_n]
        # 把dimuse的位置进行赋值(言下之意就是dim_ignored就保持不变 只更新dim_used的位置)
        # 输出完整的99dim
        ang_out[:, :, dim_used] = ang_out_all

        # reshape为2维 第一位-1自动计算 第二维就是99个位置信息 再加将99 分成33 * 3
        ang_out_euler = ang_out.reshape([-1, 99]).reshape([-1, 3])
        # 取GT
        ang_gt_euler = ang_h36[:, in_n:in_n + out_n].reshape([-1, 99]).reshape([-1, 3])

        import utils.data_utils as data_utils
        # 预测的欧拉角
        ang_out_euler = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(ang_out_euler))
        ang_out_euler = ang_out_euler.view(-1, out_n, 99)
        # gt欧拉角
        ang_gt_euler = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(ang_gt_euler))
        ang_gt_euler = ang_gt_euler.view(-1, out_n, 99)

        # 计算误差和 最后输出一个向量
        # 为什么会有归一化(将3维将到2维) 论文公式里没有啊norm就是计算向量模长
        # sum dim=0 将2维降到1维
        eulererr_ang_seq = torch.sum(torch.norm(ang_out_euler - ang_gt_euler, dim=2), dim=0)
        # 这里的m_ang_seq 是一个列表 这里的+=操作就是列表里每个位置的元素对应相加(+=)
        m_ang_seq += eulererr_ang_seq.cpu().data.numpy()

    ret = {}
    m_ang_h36 = m_ang_seq / n  # 除总样本数 n = batch_size * batch_num
    for j in range(out_n):
        ret["#{:d}".format(titles[j])] = m_ang_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()  # 这里还带打印参数设置
    main(option)
