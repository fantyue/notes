# **CRF**
**马尔科夫随机场**：随机场是由若干个位置组成的整体(无向图)，当给每一个位置按某种随机分布赋予一个值后，其整体就成为随机场。而马尔科夫随机场是随机场的特例，其假设随机场中的某一个位置只与它相邻的位置的赋值有关，和不相邻位置的赋值无关。

**CRF**:设$X$和$Y$是随机变量，$P(Y|X)$是给定$X$时$Y$的条件概率分布，若随机变量$Y$构成一个马尔科夫随机场，则称条件概率分布$P(Y|X)$为条件随机场。

**crf定义了两种特征函数：**   
第一类是定义在$Y$节点上的特征函数，这类特征函数只和当前节点有关，记为：
$$
s_l(y_i,x,i), l=1,2,...,L \tag{1}
$$
第二类定义在$Y$上下文上的特征函数，这类特征函数只和当前节点和前一个节点有关,记为：
$$t_k(y_{i-1},y_i,x,i), l=1,2,...,L$$

### **linear-crf参数化形式**  
$$
P(Y|X)=\frac{exp(\sum \limits_{i,k}\lambda_k t_k(y_{i-1},y_i,x,i)+\sum \limits_{i,l}\mu_l s_l(y_i,x,i))}{Z(x)}\tag{2}
$$
其中，$Z(x)$为规范化因子：
$$
Z(x)=\sum \limits_y exp(\sum \limits_{i,k}\lambda_k t_k(y_{i-1},y_i,x,i)+\sum \limits_{i,l}\mu_l s_l(y_i,x,i)) \tag{3}
$$

### **linear-crf简化形式**  
将两种特征函数合并，统一使用$f_k(y_{i-1},y_i,x,i)$
形式表示：
$$
f_k(y_{i-1},y_i,x,i)=
\begin{cases}
t_k(y_{i-1},y_i,x,i) & k=1,2,\dots,K_1\\
s_l(y_i,x,i) & k=K_1+l,l=1,2,\dots,K_2 
\end{cases} \tag{4}
$$
同时特征函数的权重也可以统一为：
$$
w_k=
\begin{cases}
\lambda_k & k=1,2,\dots,K_1\\
\mu_l & k=K_1+l,l=1,2,\dots,K_2
\end{cases} \tag{5}
$$

### **linear-crf的三个基本问题**
```
1. 评估:给定linear-crf的概率分布$P(y|x)$,在给定输入序列$X$和输出序列$Y$，计算条件概率$P(y_i|x)$和$P(y_{i-1}，y_i|x)$
2. 学习
3. 解码
```

### **linear-crf模型学习**
linear-crf的条件概率分布如$(2)$,条件分布$P(Y|X)$的对数似然如下:


# **NN_CRF**
**NNCRF类**
```
class NNCRF(nn.Module):
    def __init__(self, tagset_size):
        super().__init__()
        self.tagset_size = tagset_size + 2
        self.hidden_dim = 768

        self.START = -2
        self.STOP = -1

        self.encoder = BertModel.from_pretrained('bert-base-chinese')
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # trans[from, to]
        self.trans = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))
        nn.init.xavier_normal(self.trans)

        self.trans[:, self.START] = -10000.0
        self.trans[self.STOP, :] = -10000.0
```

### **概率计算**

```
    def _calc_partition(self, feats, mask):
        """
        :param feats: (batch_size, seq_len, tagset_size)
        :param mask: (batch_size, seq_len)
        :return:
        """
        b, s, t = feats.size(0), feats.size(1), feats.size(2)
        assert (t == self.tagset_size)
        mask = mask.transpose(0, 1)
        feats = feats.transpose(0, 1).view(s, b, 1, t).expand(s, b, t, t)
        scores = feats + self.trans.view(1, t, t)

        seq_iter = enumerate(scores)
        _, inivalue = next(seq_iter)
        # partition:[batch_size, 1, tagset_size]
        partition = inivalue[:, self.START, :].view(b, t, 1)

        for idx, cur_value in seq_iter:
            # cur_value: [batch_size, tagset_size, tagset_size]
            # 代表 from -> to tag的分值
            # partition保存的是上一步的to tag的路径概率和
            # 上一步的to是当前步的from，需要转置二三维
            cur_value = cur_value + partition.view(b, t, 1)
            # 当前步的路径概率和为torch.exp(cur_value),再在from维度求和，最后取对数
            # cur_value:[batch_size, tagset_size]
            cur_value = torch.logsumexp(cur_value, dim=1)

            # 只更新没有mask位置的partition
            # 拿到当前位置的mask id
            cur_mask_idx = mask[idx].view(b, 1, 1).expand(b, t, 1)
            # 拿到没有被mask位置的值
            # unmasked_value = cur_value.masked_select(cur_mask_idx)
            # cur_mask_idx = cur_mask_idx.view(b ,t , 1)
            # partition.masked_scatter_(cur_mask_idx, unmasked_value)
            partition = partition.mul((1 - cur_mask_idx).byte()) + cur_value.view(b, t, 1).mul(cur_mask_idx)
        # 添加stop tag 分值
        last_values = partition.expand(b, t, t) + self.trans.view(1, t, t)
        last_values = torch.logsumexp(last_values, 1)
        final_partition = last_values[:, self.STOP]

        return final_partition.sum()
```
### **viterbi解码**

```
    def _viterbi(self, feats, mask):
        # b:batch_size, s:seq_len, t:tagset_size
        b, s, t = feats.size(0), feats.size(1), feats.size(2)
        assert (t == self.tagset_size)

        mask_length = torch.sum(mask.int(), dim=1).view(b, 1).long()
        mask = mask.transpose(1, 0).contiguous()
        mask = (1 - mask).byte()

        trans = self.trans.view(1, t, t)

        feats = feats.transpose(1, 0).contiguous().view(s, b, 1, t).expand(s, b, t, t)
        scores = feats + trans

        # back_points保存从第2位到第len(sentence)位的前一步最大tag
        back_points = list()
        partition_history = list()

        # batch_size * tagset_size
        partition = scores[0, :, self.START, :]
        partition_history.append(partition.view(1, b, t))

        for i in range(1, s):
            cur_value = scores[i, :, :, :]
            # trans: from -> to
            cur_value = cur_value + partition.view(b, t, 1).expand(b, t, t)
            partition, pre_idx = torch.max(cur_value, 1)
            back_points.append(pre_idx)
            pre_idx.masked_fill_(mask[i].view(b, 1).expand(b, t), 0)
            partition_history.append(partition.view(1, b, t))
        partition_history = torch.cat(partition_history, dim=0).transpose(0, 1)

        # mask
        last_position = mask_length.view(b, 1, 1).expand(b, 1, t) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(b, t, 1)
        last_values = last_partition + trans
        path_score, last_bp = torch.max(last_values, 1)

        pad_zero = torch.zeros(b, t).long()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(s, b, t).transpose(0, 1)

        pointer = last_bp[:, self.STOP]
        back_points.scatter_(1, last_position, pointer.view(b, 1, 1).expand(b, 1, t))
        back_points = back_points.transpose(0, 1)

        decode_idx = torch.LongTensor(s, b)
        decode_idx[-1] = pointer
        # pdb.set_trace()
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.view(b, 1))
            decode_idx[idx] = pointer.view(b)

        # mask
        decode_idx = decode_idx.masked_fill(mask, -1).transpose(0, 1)
        # decode_idx = decode_idx.transpose(0, 1)
        return path_score, decode_idx

```

### **模型训练**
