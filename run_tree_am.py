import os
import random
import torch
from torch import optim
import pprint as pp
import numpy as np
import arguments
from train import *
from eval import *
from nets.TreeAttentionModel import *

wart_start = False

if __name__ == '__main__':
    argParser = arguments.get_arg_parser("tree")
    args = argParser.parse_args()
    t.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.cuda = not args.cpu and torch.cuda.is_available()
    args.run_name = "run_{}".format(time.strftime("%Y%m%dT%H%M%S"))
    args.save_dir = os.path.join(
        os.getcwd(),
        args.output_dir,
        "tree_{}".format(args.node_size-1),
        args.run_name
    )
    # Pretty print the run args
    pp.pprint(vars(args))
    # mkdir outputs
    os.makedirs(args.save_dir)
    if t.cuda.is_available():
        args.DEVICE = t.device('cuda')
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        map_location = None
    else:
        args.DEVICE = t.device('cpu')
        map_location = t.device('cpu')

    # 构建两个相同结构的net,参数定期同步
    RolloutNet = AttentionModel(args)
    RolloutNet = RolloutNet.to(args.DEVICE)
    if wart_start:
        RolloutNet.load_state_dict(t.load('tree_am/epoch0-i1299-dis_8.94567.pt', map_location=map_location))
    baseNet = AttentionModel(args)
    baseNet = baseNet.to(args.DEVICE)
    baseNet.load_state_dict(RolloutNet.state_dict())

    is_train = True  # 是
    if is_train:
        if args.optimizer == 'adam':
            # [{'params': model.parameters(), 'lr': opts.lr_model}]
            optimizer = optim.Adam(RolloutNet.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(RolloutNet.parameters(), lr=args.lr)
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(RolloutNet.parameters(), lr=args.lr)
        else:
            raise ValueError('optimizer undefined: ', args.optimizer)
        # 训练部分
        train(args, optimizer, baseNet, RolloutNet)
    else:
        # 测试部分
        evaluate(args, RolloutNet, map_location)
