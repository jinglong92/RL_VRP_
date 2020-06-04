import time
import torch as t


# 测试部分
def evaluate(args, RolloutNet, map_location):
    DEVICE = args.DEVICE
    RolloutNet.load_state_dict(t.load('tree_am/AM_VRP20.pt', map_location=map_location))
    # 按照greedy策略测试
    if args.rollout_method == 'greedy':
        tS = t.rand(args.batch_size * args.test_times, args.node_size, 2)  # 坐标0~1之间
        tD = np.random.randint(1, 10, size=(args.batch_size * args.test_times, args.node_size, 1))  # 所有客户的需求
        tD = t.LongTensor(tD)
        tD[:, 0, 0] = 0  # 仓库点的需求为0
        sum_dis = t.zeros(1).to(DEVICE)

        sum_clock = 0  # 记录生成解的总时间
        for i in range(args.test_times):
            t.cuda.empty_cache()
            s = tS[i * args.batch_size: (i + 1) * args.batch_size]
            d = tD[i * args.batch_size: (i + 1) * args.batch_size]
            s = s.to(DEVICE)
            d = d.to(DEVICE)
            clock1 = time.time()
            children_seq, father_seq, pro, dis = RolloutNet(s, d, args.capacity, 'greedy', args.DEVICE)
            clock2 = time.time()
            deta_clock = clock2 - clock1
            sum_clock = sum_clock + deta_clock

            print("i:{}, mean_dis:{}, deta_clock:{}".format(i, t.mean(dis), deta_clock))
            sum_dis = sum_dis + t.mean(dis)
        mean_dis = sum_dis / args.test_times
        mean_clock = sum_clock / args.test_times
        print("mean_dis:{}, mean_clock:{}".format(mean_dis, mean_clock))

    # 按照sampling策略测试
    else:
        tS = t.rand(args.test_times, args.node_size, 2)  # 坐标0~1之间
        tD = np.random.randint(1, 10, size=(args.test_times, args.node_size, 1))  # 所有客户的需求
        tD = t.LongTensor(tD)
        tD[:, 0, 0] = 0  # 仓库点的需求为0

        all_repeat_size = 1280
        num_batch_repeat = all_repeat_size // args.batch_size

        sum_dis = t.zeros(1).to(DEVICE)
        sum_clock = 0  # 记录生成解的总时间
        for i in range(args.test_times):
            t.cuda.empty_cache()
            available_seq = []
            available_dis = []
            deta_clock = 0
            for _ in range(num_batch_repeat):
                s = tS[i].repeat(args.batch_size, 1, 1).to(DEVICE)
                d = tD[i].repeat(args.batch_size, 1, 1).to(DEVICE)

                clock1 = time.time()
                children_seq, father_seq, pro, dis = RolloutNet(s, d, args.capacity, 'sampling', args.DEVICE)
                clock2 = time.time()

                mini_deta_clock = clock2 - clock1
                deta_clock = deta_clock + mini_deta_clock
                for j in range(args.batch_size):
                    available_seq.append(seq[j])
                    available_dis.append(dis[j])

            available_seq = t.stack(available_seq)
            available_dis = t.stack(available_dis)

            mindis, mindis_index = t.min(available_dis, 0)
            sum_dis = sum_dis + mindis
            sum_clock = sum_clock + deta_clock
            print("i:{}, mindis:{}, deta_clock:{}".format(i, mindis, deta_clock))
        mean_dis = sum_dis / args.test_times
        mean_clock = sum_clock / args.test_times
        print("mean_dis:{}, mean_clock:{}".format(mean_dis.item(), mean_clock))
