import math

import torch.nn as nn
import torch as t
import torch.nn.functional as F


class AttentionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding_size = args.embedding_size
        self.batch = args.batch_size
        self.node_size = args.node_size
        self.M = args.M
        self.dk = self.embedding_size // self.M  # 多头注意力中每一头的维度
        self.dv = self.embedding_size // self.M  # 多头注意力中每一头的维度
        self.dff = self.embedding_size * 4
        self.C = args.C
        self.N = 3

        self.embedding = nn.Linear(3, self.embedding_size)  # 用于客户点的坐标加容量需求的embedding:[x, y, Cap]
        self.embedding_p = nn.Linear(2, self.embedding_size)  # 用于仓库点的坐标embedding [x, y]

        self.dense_q = nn.ModuleList([nn.Linear(self.embedding_size, self.dk, bias=False) for _ in range(self.N * self.M)])
        self.dense_v = nn.ModuleList([nn.Linear(self.embedding_size, self.dv, bias=False) for _ in range(self.N * self.M)])
        self.dense_k = nn.ModuleList([nn.Linear(self.embedding_size, self.dk, bias=False) for _ in range(self.N * self.M)])
        self.dense_o = nn.ModuleList([nn.Linear(self.dv, self.embedding_size, bias=False) for _ in range(self.N * self.M)])

        self.w_bn = nn.ParameterList([nn.Parameter(t.randn(self.embedding_size)) for _ in range(self.N)])
        self.b_bn = nn.ParameterList([nn.Parameter(t.randn(self.embedding_size)) for _ in range(self.N)])

        self.dense_ff_0 = nn.ModuleList([nn.Linear(self.embedding_size, self.dff) for _ in range(self.N)])
        self.dense_ff_1 = nn.ModuleList([nn.Linear(self.dff, self.embedding_size) for _ in range(self.N)])

        self.softmax = nn.Softmax(2)

        self.wq1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.wk1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.wv1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.w1 = nn.Linear(self.embedding_size, self.embedding_size)

        self.wq2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.wk2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.wv2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.w2 = nn.Linear(self.embedding_size, self.embedding_size)

        self.wq3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.wk3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.wv3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.w3 = nn.Linear(self.embedding_size, self.embedding_size)

        self.wq = nn.Linear(self.embedding_size * 2 + 1, self.embedding_size)
        self.wk = nn.Linear(self.embedding_size, self.embedding_size)
        self.wv = nn.Linear(self.embedding_size, self.embedding_size)
        self.w = nn.Linear(self.embedding_size, self.embedding_size)

        self.q = nn.Linear(self.embedding_size, self.embedding_size)
        self.k = nn.Linear(self.embedding_size, self.embedding_size)

        self.fw1 = nn.Linear(self.embedding_size, self.dff)
        self.fb1 = nn.Linear(self.dff, self.embedding_size)

        self.fw2 = nn.Linear(self.embedding_size, self.dff)
        self.fb2 = nn.Linear(self.dff, self.embedding_size)

        self.fw3 = nn.Linear(self.embedding_size, self.dff)
        self.fb3 = nn.Linear(self.dff, self.embedding_size)

        # Batch Normalization(BN)
        self.BN11 = nn.BatchNorm1d(self.embedding_size)
        self.BN12 = nn.BatchNorm1d(self.embedding_size)
        self.BN21 = nn.BatchNorm1d(self.embedding_size)
        self.BN22 = nn.BatchNorm1d(self.embedding_size)
        self.BN31 = nn.BatchNorm1d(self.embedding_size)
        self.BN32 = nn.BatchNorm1d(self.embedding_size)

    def cal_distance(self, s):
        # s :[batch x seq_len x 2]
        s1 = t.unsqueeze(s, dim=1)  # 在s中指定位置N加上一个维数为1的维度
        s1 = s1.expand(self.batch, self.node_size, self.node_size, 2)  # 返回当前张量在某维扩展更大后的张量
        s2 = t.unsqueeze(s, dim=2)
        s2 = s2.expand(self.batch, self.node_size, self.node_size, 2)
        ss = s1 - s2  # 坐标差
        dis = t.norm(ss, 2, dim=3, keepdim=True)  # dis表示任意两点间的距离 (batch, node_size, node_size, 1)
        return dis

    def single_head_attention(self, x, n):
        h = 0
        for m in range(self.M):
            q = self.dense_q[n * self.M + m](x)
            v = self.dense_v[n * self.M + m](x)
            k = self.dense_k[n * self.M + m](x)
            k_t = t.transpose(k, 1, 2)
            u = t.bmm(q, k_t) / math.sqrt(self.dk)
            s = self.softmax(u)
            tt = t.bmm(s, v)
            h += self.dense_o[n * self.M + m](tt)
        return h

    def feed_forward(self, x, n):
        h_0 = F.relu(self.dense_ff_0[n](x))
        h_1 = self.dense_ff_1[n](h_0)
        return h_1

    def multi_head_attention(self, h, x):
        self.bn = nn.BatchNorm1d(x.size()[1], affine=False)
        for n in range(self.N):
            h_mha = self.single_head_attention(h, n)
            h_bn = self.w_bn[n] * self.bn(h + h_mha) + self.b_bn[n]
            h_ff = self.feed_forward(h_bn, n)
            h = self.w_bn[n] * self.bn(h_bn + h_ff) + self.b_bn[n]

        return h

    # s坐标，d需求，capacity初始容量
    def forward(self, s, d, capacity, train, DEVICE):
        mask_size = t.LongTensor(self.batch).to(DEVICE)  # 用于标号，方便后面两点间的距离计算
        for i in range(self.batch):
            mask_size[i] = self.node_size * i  # depot对应的编号
        # 生成距离关系，类似distance matrix
        dis = self.cal_distance(s)
        # 定义各变量Tensor
        pro = t.FloatTensor(self.batch, self.node_size * 2).to(DEVICE)  # 每个点被选取时的选取概率,将其连乘可得到选取整个路径的概率
        seq = t.LongTensor(self.batch, self.node_size * 2).to(DEVICE)  # 选取的序列
        vehicle_index = t.LongTensor(self.batch).to(DEVICE)  # 当前车辆所在的点
        tag = t.ones(self.batch * self.node_size).to(DEVICE)
        distance = t.zeros(self.batch).to(DEVICE)  # 总距离
        rest_cap = t.LongTensor(self.batch, 1, 1).to(DEVICE)  # 车的剩余容量
        demand = t.LongTensor(self.batch, self.node_size).to(DEVICE)  # 客户需求 [batch, batch]

        # node input 初始化：初始容量，负荷需求，初始位置
        rest_cap[:, 0, 0] = capacity  # 初始化车的初始容量
        demand[:, :] = d[:, :, 0]  # 需求
        vehicle_index[:] = 0
        feature = t.cat([s, d.float()], dim=2)  # [batch x seq_len x 3] 坐标与容量需求拼接

        ################################ encoder #####################################
        # node embedding
        node = self.embedding(feature)  # 客户点embedding坐标加容量需求[batch x seq_len x 3] * [3, embedding_size]
        # todo depot 不需要单独处理？
        node[:, 0, :] = self.embedding_p(s[:, 0, :])  # 仓库点只embedding坐标
        node_embedding = node  # [batch x seq_len x embedding_dim]

        # x = self.encoder(feature, s, d, capacity)  # feature
        #######################################################################
        x_compare = self.multi_head_attention(node, feature)

        # 第一个MHA attention layer, 8层，
        query1 = self.wq1(node)
        query1 = t.unsqueeze(query1, dim=2)
        query1 = query1.expand(self.batch, self.node_size, self.node_size, self.embedding_size)
        key1 = self.wk1(node)
        key1 = t.unsqueeze(key1, dim=1)
        key1 = key1.expand(self.batch, self.node_size, self.node_size, self.embedding_size)
        value1 = self.wv1(node)
        value1 = t.unsqueeze(value1, dim=1)
        value1 = value1.expand(self.batch, self.node_size, self.node_size, self.embedding_size)
        x = query1 * key1
        x = x.view(self.batch, self.node_size, self.node_size, self.M, -1)
        x = t.sum(x, dim=4)  # u=q^T x k
        x = x / (self.dk ** 0.5)
        x = F.softmax(x, dim=2)
        # softmax() * V
        x = t.unsqueeze(x, dim=4)
        x = x.expand(self.batch, self.node_size, self.node_size, self.M, 16)
        x = x.contiguous()
        x = x.view(self.batch, self.node_size, self.node_size, -1)
        Z = x * value1
        Z = t.sum(Z, dim=2)  # MHA :(batch, node_size, embedding_size)
        MHA_l = self.w1(Z)  # 输出得到MHA的Z, w1为attention的权重
        ########## MHA layer1: Add&Norm, 残差连接(Skip connection)和批归一化(Batch Normalization, BN)  ###########
        ######## h_i_hat, skip-connections #########
        x = node_embedding + MHA_l  # h_i^{(l-1)} + MHA_i^l(h_1^{(l-1)}, \dots, h_n^{(l-1)})
        # 第一个BN
        x = x.permute(0, 2, 1)
        x = self.BN11(x)
        x = x.permute(0, 2, 1)
        # x = t.tanh(x)
        # FF
        x1 = self.fw1(x)
        x1 = F.relu(x1)
        x1 = self.fb1(x1)
        ######## h_i^(l) #########
        x = x + x1
        # 第二个BN
        x = x.permute(0, 2, 1)
        x = self.BN12(x)
        x = x.permute(0, 2, 1)

        x1 = x  # h_i^(l) n=1
        #######################################################################
        # 第2个MHA attention layer, 8层
        query2 = self.wq2(x)
        query2 = t.unsqueeze(query2, dim=2)
        query2 = query2.expand(self.batch, self.node_size, self.node_size, self.embedding_size)
        key2 = self.wk2(x)
        key2 = t.unsqueeze(key2, dim=1)
        key2 = key2.expand(self.batch, self.node_size, self.node_size, self.embedding_size)
        value2 = self.wv2(x)
        value2 = t.unsqueeze(value2, dim=1)
        value2 = value2.expand(self.batch, self.node_size, self.node_size, self.embedding_size)
        x = query2 * key2
        x = x.view(self.batch, self.node_size, self.node_size, self.M, -1)
        x = t.sum(x, dim=4)
        x = x / (self.dk ** 0.5)
        x = F.softmax(x, dim=2)
        x = t.unsqueeze(x, dim=4)
        x = x.expand(self.batch, self.node_size, self.node_size, self.M, 16)
        x = x.contiguous()
        x = x.view(self.batch, self.node_size, self.node_size, -1)
        Z = x * value2
        Z = t.sum(Z, dim=2)  # MHA :(batch, node_size, embedding_size)
        MHA_l = self.w2(Z)  # 输出得到MHA的Z, w1为attention的权重
        ########## MHA layer2: Add&Norm, 残差连接(Skip connection)和批归一化(Batch Normalization, BN)  ###########
        ######## h_i_hat #########
        x = MHA_l + x1
        # 第一个BN
        x = x.permute(0, 2, 1)
        x = self.BN21(x)
        x = x.permute(0, 2, 1)
        # FF
        x1 = self.fw2(x)
        x1 = F.relu(x1)
        x1 = self.fb2(x1)
        ######## h_i^(l) #########
        x = x + x1
        # 第二个BN
        x = x.permute(0, 2, 1)
        x = self.BN22(x)
        x = x.permute(0, 2, 1)

        x1 = x  # h_i^(l) n=2
        #######################################################################
        # 第三层MHA
        query3 = self.wq3(x)
        query3 = t.unsqueeze(query3, dim=2)
        query3 = query3.expand(self.batch, self.node_size, self.node_size, self.embedding_size)
        key3 = self.wk3(x)
        key3 = t.unsqueeze(key3, dim=1)
        key3 = key3.expand(self.batch, self.node_size, self.node_size, self.embedding_size)
        value3 = self.wv3(x)
        value3 = t.unsqueeze(value3, dim=1)
        value3 = value3.expand(self.batch, self.node_size, self.node_size, self.embedding_size)
        x = query3 * key3
        x = x.view(self.batch, self.node_size, self.node_size, self.M, -1)
        x = t.sum(x, dim=4)
        x = x / (self.dk ** 0.5)
        x = F.softmax(x, dim=2)
        x = t.unsqueeze(x, dim=4)
        x = x.expand(self.batch, self.node_size, self.node_size, self.M, 16)
        x = x.contiguous()
        x = x.view(self.batch, self.node_size, self.node_size, -1)
        Z = x * value3
        Z = t.sum(Z, dim=2)
        MHA_l = self.w3(Z)
        ########## MHA layer3: Add&Norm, 残差连接(Skip connection)和批归一化(Batch Normalization, BN)  ###########
        ######## h_i_hat #########
        x = MHA_l + x1
        #####################
        # 第一个BN
        x = x.permute(0, 2, 1)
        x = self.BN31(x)
        x = x.permute(0, 2, 1)
        # FF
        x1 = self.fw3(x)
        x1 = F.relu(x1)
        x1 = self.fb3(x1)
        ######## h_i^(l) #########
        x = x + x1
        # 第二个BN
        x = x.permute(0, 2, 1)
        x = self.BN32(x)
        x = x.permute(0, 2, 1)  # h_i^(l) n=3 (batch, node_size, embedding_size)
        x = x.contiguous()

        # graph embedding
        avg = t.mean(x, dim=1)  # 最后将所有节点的嵌入信息取平均得到整个图的嵌入信息，(batch, embedding_size)

        ################################# decoder ######################################
        for i in range(self.node_size * 2):  # decoder输出序列的长度不超过node_size * 2
            flag = t.sum(demand, dim=1)  # demand:(batch, self.node_size)
            f1 = t.nonzero(flag > 0).view(-1)  # 取得需求不全为0的batch号
            f2 = t.nonzero(flag == 0).view(-1)  # 取得需求全为0的batch号

            if f1.size()[0] == 0:  # batch所有需求均为0
                pro[:, i:] = 1  # pro:(batch, node_size*2)
                seq[:, i:] = 0  # swq:(batch, node_size*2)
                temp = dis.view(-1, self.node_size, 1)[
                    vehicle_index + mask_size]  # dis:任意两点间的距离 (batch, node_size, node_size, 1) temp:(batch, node_size,1)
                distance = distance + temp.view(-1)[mask_size]  # 加上当前点到仓库的距离
                break

            ### 1.开始构造：Context embedding ###
            ind = vehicle_index + mask_size
            tag[ind] = 0  # tag:(batch*node_size)
            start = x.view(-1, self.embedding_size)[ind]  # (batch, embedding_size)，每个batch中选出一个节点

            end = rest_cap[:, :, 0]  # (batch, 1)
            end = end.float()  # 车上剩余容量

            # 1.1结合图embedding，当前点embedding，车剩余容量: (batch,embedding_size*2 + 1)_
            graph = t.cat([avg, start, end], dim=1)
            query = self.wq(graph)  # (batch, embedding_size)
            query = t.unsqueeze(query, dim=1)
            query = query.expand(self.batch, self.node_size, self.embedding_size)
            key = self.wk(x)
            value = self.wv(x)
            temp = query * key
            temp = temp.view(self.batch, self.node_size, self.M, -1)
            temp = t.sum(temp, dim=3)  # (batch, node_size, M)
            temp = temp / (self.dk ** 0.5)
            # 1.2 mask: set u_{(c)j} = -inf
            mask = tag.view(self.batch, -1, 1) < 0.5  # 访问过的点tag=0
            mask1 = demand.view(self.batch, self.node_size, 1) > rest_cap.expand(self.batch, self.node_size,
                                                                             1)  # 客户需求大于车剩余容量的点

            flag = t.nonzero(vehicle_index).view(-1)  # 在batch中取得当前车不在仓库点的batch号
            mask = mask + mask1  # mask:(batch x node_size x 1)
            mask = mask > 0
            mask[f2, 0, 0] = 0  # 需求全为0则使车一直在仓库
            if flag.size()[0] > 0:  # 将有车不在仓库的batch的仓库点开放
                mask[flag, 0, 0] = 0
            mask = mask.expand(self.batch, self.node_size, self.M)  # expand for MHA

            # 1.3 将mask为True的节点置为-inf
            temp.masked_fill_(mask, -float('inf'))
            temp = F.softmax(temp, dim=1)
            temp = t.unsqueeze(temp, dim=3)
            temp = temp.expand(self.batch, self.node_size, self.M, 16)
            temp = temp.contiguous()
            temp = temp.view(self.batch, self.node_size, -1)
            temp = temp * value
            Z = t.sum(temp, dim=1)
            hc_N = self.w(Z)  # hc,(batch,embedding_size) # 输出得到MHA的Z, self.w为attention的权重

            ### 2.Calculation of log-probabilities ###
            # 2.1 add one final decoder layer with a single attention head(M=1, dk = dh)
            query = self.q(hc_N)
            key = self.k(x)  # (batch, node_size, embedding_size)
            query = t.unsqueeze(query, dim=1)  # (batch, 1 ,embedding_size)
            query = query.expand(self.batch, self.node_size, self.embedding_size)  # (batch, node_size, embedding_size)
            temp = query * key
            temp = t.sum(temp, dim=2)
            temp = temp / (self.dk ** 0.5)
            # 2.2 clip the result before masking using tanh
            temp = t.tanh(temp) * self.C  # (batch, node_size)
            # 2.3 将mask为True的节点置为-inf
            mask = mask[:, :, 0]
            temp.masked_fill_(mask, -float('inf'))
            # 2.4 compute the final probability vector p using softmax
            p = F.softmax(temp, dim=1)  # 得到选取每个点时所有点可能被选择的概率
            # 2.5 对需求不全为0的batch，进行抽样（按照概率p抽取一个点）或直接选择概率最大的点
            indexx = t.LongTensor(self.batch).to(DEVICE)
            if train == 'sampling':
                indexx[f1] = t.multinomial(p[f1], 1)[:, 0]  # 按sampling策略选点
            else:
                # 语法：(t.max(p[f1], dim=1)[0] 为value, ()[1]为index
                indexx[f1] = (t.max(p[f1], dim=1)[1])  # 按greedy策略选点

            indexx[f2] = 0
            p = p.view(-1)
            pro[:, i] = p[indexx + mask_size]
            pro[f2, i] = 1
            rest_cap = rest_cap - (demand.view(-1)[indexx + mask_size]).view(self.batch, 1, 1)  # 车的剩余容量
            demand = demand.view(-1)
            demand[indexx + mask_size] = 0
            demand = demand.view(self.batch, self.node_size)

            temp = dis.view(-1, self.node_size, 1)[vehicle_index + mask_size]
            distance = distance + temp.view(-1)[indexx + mask_size]

            mask3 = indexx == 0
            mask3 = mask3.view(self.batch, 1, 1)
            rest_cap.masked_fill_(mask3, capacity)  # 车回到仓库将容量设为初始值

            vehicle_index = indexx
            seq[:, i] = vehicle_index[:]

        if train == 'greedy':
            seq = seq.detach()
            pro = pro.detach()
            distance = distance.detach()

        return seq, pro, distance  # 被选取的点序列,每个点被选取时的选取概率,这些序列的总路径长度
