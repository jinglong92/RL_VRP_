import os
import random
import torch
from torch import optim

import numpy as np
from conf import arguments
from train import *
from eval import *
from nets.TreeAttentionModel import *

t.manual_seed(111)
random.seed(111)
np.random.seed(111)

wart_start = False

if __name__ == '__main__':
    argParser = arguments.get_arg_parser("tree")
    args = argParser.parse_args()
    args.cuda = not args.cpu and torch.cuda.is_available()
    if t.cuda.is_available():
        DEVICE = t.device('cuda')
        map_location = None
    else:
        DEVICE = t.device('cpu')
        map_location = t.device('cpu')
    args.DEVICE = DEVICE

    save_dir = os.path.join(os.getcwd(), args.output_dir)
    # 构建两个相同结构的net,参数定期同步
    RolloutNet = AttentionModel(args)
    RolloutNet = RolloutNet.to(DEVICE)
    if wart_start:
        RolloutNet.load_state_dict(t.load('tree_am/epoch0-i1299-dis_8.94567.pt', map_location=map_location))
    baseNet = AttentionModel(args)
    baseNet = baseNet.to(DEVICE)
    baseNet.load_state_dict(RolloutNet.state_dict())

    is_train = True  # 是
    if is_train:
        if args.optimizer == 'adam':
            optimizer = optim.Adam(RolloutNet.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(RolloutNet.parameters(), lr=args.lr)
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(RolloutNet.parameters(), lr=args.lr)
        else:
            raise ValueError('optimizer undefined: ', args.optimizer)
        # 训练部分
        train(args, optimizer, baseNet, RolloutNet, save_dir)
    else:
        # 测试部分
        evaluate(args, RolloutNet, map_location)
