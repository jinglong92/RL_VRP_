from data_utils.data_utils import data_gen
from scipy.stats import ttest_rel
import time
import torch as t
import torch.nn as nn


def train(args, opt, baseNet, RolloutNet):
    DEVICE = args.DEVICE
    tS, tD, S, D = data_gen(args.batch_size, args.test2save_times, args.node_size, args.inner_times)
    # # print model's state_dict, 学习的权重和偏执系数, 卷积层和全连接层的参数
    # print('Model.state_dict:')
    # for param_tensor in RolloutNet.state_dict():
    #     print(param_tensor, '\t', RolloutNet.state_dict()[param_tensor].size())  # 打印 key value字典
    #
    # # print optimizer's state_dict, 包含state和param_groups的字典对象
    # print('Optimizer,s state_dict:')
    # for var_name in opt.state_dict():
    #     print(var_name, '\t', opt.state_dict()[var_name])

    for epoch in range(args.epochs):
        for i in range(args.inner_times):
            t.cuda.empty_cache()
            s = S[i * args.batch_size: (i + 1) * args.batch_size]  # [batch x seq_len x 2]
            d = D[i * args.batch_size: (i + 1) * args.batch_size]  # [batch x seq_len x 1]
            s = s.to(DEVICE)  # s 传到DEVICE上执行
            d = d.to(DEVICE)  # d 传到DEVICE上执行

            t1 = time.time()
            # 被选取的点序列,每个点被选取时的选取概率,这些序列的总路径长度
            seq2, pro2, dis2 = baseNet(s, d, args.capacity, 'greedy', DEVICE)  # baseline
            seq1, pro1, dis1 = RolloutNet(s, d, args.capacity, 'sampling', DEVICE)  # samplingRollout
            t2 = time.time()
            # print('nn_output_time={}'.format(t2 - t1))
            ######################### forward + backward + optimize ##############################
            # optimizer.zero_grad()把梯度置零，也就是把loss关于weight的导数变成0.
            opt.zero_grad()
            # 带baseline的policy gradient训练算法, dis2作为baseline
            log_prob = t.sum(t.log(pro1), dim=1)
            L_pai = dis1 - dis2  # advantage reward(优势函数)
            L_pai_detached = L_pai.detach()  # 创建一个新的tensor,新的tensor与之前的共享data,但是不具有梯度
            loss = t.sum(L_pai_detached * log_prob) / args.batch_size  # 最终损失函数
            # 反向传播求梯度
            loss.backward()
            # 梯度爆炸解决方案——梯度截断（gradient clip norm）
            nn.utils.clip_grad_norm_(RolloutNet.parameters(), 1)
            opt.step()  # Performs a single optimization step (parameter update)
            print('epoch={}, i={}, mean_dis1={}, mean_dis2={}'.format(epoch, i, t.mean(dis1), t.mean(dis2)))
            # ,'disloss:',t.mean((dis1-dis2)*(dis1-dis2)), t.mean(t.abs(dis1-dis2)), nan)

            ################# OneSidedPairedTTest: 配对样本T检验标准分T=(x−μ)/(s/sqrt(n)) #####################
            # (paired t-检验,当前Sampling的解效果是否显著好于greedy的解效果,如是,则更新使用greedy策略作为baseline的net2参数)
            if (dis1.mean() - dis2.mean()) < 0:
                t_statistic, p_value = ttest_rel(dis1.cpu().numpy(), dis2.cpu().numpy())
                p_value = p_value / 2
                assert t_statistic < 0, "T-statistic should be negative"
                if p_value < args.p_threshold:  # If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%,
                    # then we reject the null hypothesis of equal averages.
                    print(' ------------- Update baseline ------------- ')
                    baseNet.load_state_dict(RolloutNet.state_dict())
            ################# 每隔100步做测试判断结果有没有改进，如果改进了则把当前模型保存下来 #####################
            if (i + 1) % args.log_interval == 0:
                length = t.zeros(1).to(DEVICE)
                for j in range(args.test2save_times):
                    t.cuda.empty_cache()
                    s = tS[j * args.batch_size: (j + 1) * args.batch_size]
                    d = tD[j * args.batch_size: (j + 1) * args.batch_size]
                    s = s.to(DEVICE)
                    d = d.to(DEVICE)
                    seq, pro, dis = RolloutNet(s, d, args.capacity, 'greedy')
                    length = length + t.mean(dis)
                mean_len = length / args.test2save_times
                if mean_len < min_length:
                    # 有改进，保存当前模型
                    t.save(RolloutNet.state_dict(), os.path.join(save_dir,
                                                                 'epoch{}-i{}-dis_{:.5f}.pt'.format(
                                                                     epoch, i, mean_len.item())))
                    min_length = mean_len
                print('min=', min_length.item(), 'length=', length.item())
