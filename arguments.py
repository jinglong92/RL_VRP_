import argparse


def get_arg_parser(title):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--max_eval_size', type=int, default=100000)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--train_proportion', type=float, default=1.0)

    parser.add_argument('--LSTM_hidden_size', type=int, default=512)
    parser.add_argument('--MLP_hidden_size', type=int, default=256)
    parser.add_argument('--param_init', type=float, default=0.1)
    parser.add_argument('--num_LSTM_layers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)

    # 每隔100步做测试判断结果有没有改进，如果改进了则把当前模型保存下来
    parser.add_argument('--log_interval', type=int, default=100)

    data_group = parser.add_argument_group('tree')
    # data_group.add_argument('--output_dir', type=str, default="tree_am")
    data_group.add_argument('--batch_size', type=int, default=512)  # 每个batch的算例数
    data_group.add_argument('--inner_times', type=int, default=2500)  # 训练中每个epoch所需的训练batch数
    data_group.add_argument('--epochs', type=int, default=100)  # 训练的epoch总数
    data_group.add_argument('--capacity', type=float, default=30000)  # 车辆的初始容量
    data_group.add_argument('--num_MLP_layers', type=int, default=2)
    data_group.add_argument('--embedding_size', type=int, default=128)
    data_group.add_argument('--attention_size', type=int, default=16)
    data_group.add_argument('--node_size', type=int, default=21)  # 节点总数
    data_group.add_argument('--n_heads', type=int, default=8)  # 多头注意力中的头数

    train_group = parser.add_argument_group('train')
    train_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
    train_group.add_argument('--lr', type=float, default=5e-5)
    train_group.add_argument('--lr_decay_steps', type=int, default=500)
    train_group.add_argument('--lr_decay_rate', type=float, default=0.95)
    train_group.add_argument('--num_epochs', type=int, default=10)
    train_group.add_argument('--dropout_rate', type=float, default=0.0)
    train_group.add_argument('--test2save_times', type=int, default=20)  # 训练过程中每次保存模型所需的测试batch数
    data_group.add_argument('--gradient_clip', type=int, default=10)  # 做softmax得到选取每个点概率前，clip the result所使用的参数
    data_group.add_argument('--p_threshold', type=int, default=0.05)  # 做t-检验更新baseline时所设置的阈值5%

    test_group = parser.add_argument_group('test')
    test_group.add_argument('--rollout_method', type=str, default='sampling')  # sampling or greedy
    test_group.add_argument('--test_times', type=int, default=10)  # 测试时所需的batch总数

    output_group = parser.add_argument_group('outputs')
    output_group.add_argument('--output_dir', default='outputs', help='Directory to write output models to')

    return parser
