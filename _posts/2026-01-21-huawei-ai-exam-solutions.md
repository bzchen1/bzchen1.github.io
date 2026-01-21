---
title: 华为AI机考真题汇总与复习
date: 2026-01-20 17:19:40 +0800
categories: [学习笔记, 华为机考]
tags: [Algorithm, AI, Python]
math: true
mermaid: true
# description: 自动生成的AI机考真题复习文档
---

> 本文内容由 Jupyter Notebook 自动转换生成。
> 记录了备战期间的算法真题与解题思路。


# code真题记录
## 0827
### KNN
`np.argsort(x,axis =1)`按照列排序（升序）后，返回索引


```python
import numpy as np
import sys
from collections import Counter
k, m, n, s = map(int, input().split())
test_x = np.zeros(n)
test_x = list(map(float, sys.stdin.readline().split()))
train_x, train_y = np.zeros((m, n)), np.zeros(m)
for i in range(m):
    line = list(map(float, sys.stdin.readline().split()))
    train_x[i] = line[:-1]
    train_y[i] = line[-1]

# 1. 计算距离
diff = np.sum((train_x - test_x) ** 2, axis=1)
# 2. 前k个
diff_sort_arg = np.argsort(diff)
topk_arg = diff_sort_arg[:k]

# 3. 投票确定
y_votes = Counter(train_y[topk_arg])
most_common = y_votes.most_common(2)
if len(most_common) == 1 or most_common[0][1] != most_common[1][1]:
    ans_label = most_common[0][0]
else:
    common_num = most_common[0][1]
    # 出现平局，需要筛选diff最近的
    for cur_dist_id in topk_arg:
        if y_votes[train_y[cur_dist_id]] == common_num:
            ans_label = train_y[cur_dist_id]
            break
print(int(ans_label), y_votes[ans_label])

```

1. 尽量写子函数（将功能分开）
2. **ai题目的技巧** `(feature,label)` -> `list`  列表的索引`i`作为该样本的编号,方便排序
3. 命名规范：features, label, dist, neighbors, top_k, votes
	- features, label = row[:-1], row[-1] 

### 决策树剪枝
1. 混淆矩阵

| 实际\预测 | 正例 | 反例 |
| -------- | ---- | ---- |
| 正例     | TP(真正例)   | FN(假反)   |
| 反例     | FP   | TN   |
	
- acc(准确率) = (TP + TN) / 总数
- precision(精确率) = TP / (TP + FP)
- recall(召回率) = TP / (TP + FN)
- F1 = 2 * (precision * recall) / (precision + recall)

2. 后剪枝：`dfs`到该节点时，计算当前截断、不截断的分数，选择较大者
3. 预剪枝：在划分节点前，计算划分前后的分数
4. 树的构造：class`Node`；树的遍历:dfs/bfs
5. 注意除0保护  `if denom == 0: return 0`

[python语言] 类与对象
1. 类的定义与实例化
	```python
	class Dog:
		species = "Canis familiaris"  # 类属性
		
		# 构造函数(初始化方法)
		def __init__(self, name):
			self.name = name  # 实例属性
		
		# 实例方法
		def bark(self):
			return f"{self.name} says woof!"

	dog1 = Dog("Buddy")
	print(dog1.bark())  # 输出: Buddy says woof!
	```

2. 类的特殊函数(在类中重写)
	- 常见特殊函数:
		- `__init__(self, ...)`：构造函数，初始化对象属性。
		- `__str__(self)`：定义对象的字符串表示，使用`print()`时调用。
		- `__len__(self)`：定义对象的长度，使用`len()`时调用。
		- `__add__(self, other)`：定义加法操作符`+`的行为。
		- `__eq__(self, other)`：定义等于操作符`==`的行为。
	```python
	class Student:
		def __init__(self,name,chinese_score,math_score,English_score):
			self.name = name
			self.chinese_score = chinese_score
			self.math_score = math_score
			self.English_score = English_score
		def __add__(self,other):
			return [self.chinese_score + other.chinese_score,
					self.math_score + other.math_score,
					self.English_score + other.English_score]
		def __str__(self):
			return f"Student Name: {self.name}, Scores - Chinese: {self.chinese_score}, Math: {self.math_score}, English: {self.English_score}"


	stu1 = Student("Alice", 85, 90, 88)
	stu2 = Student("Bob", 78, 82, 80)
	total_scores = stu1 + stu2
	print(total_scores)  # 输出: [163, 172, 168]	
	```



```python
# 题解

class Node:
    def __init__(self, l, r, f, th, label):
        self.l = l
        self.r = r
        self.f = f
        self.th = th
        self.label = label


class Result:
    def __init__(self, tp, fp, tn, fn):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

    def __add__(self, other):
        res = Result(0,0,0,0)
        res.tp = self.tp + other.tp
        res.fp = self.fp + other.fp
        res.tn = self.tn + other.tn
        res.fn = self.fn + other.fn
        return res

    def cal(self, data, label):
        for x in data:
            gt = x[-1]
            if gt == 1:
                if label == 1:
                    self.tp += 1
                else:
                    self.fn += 1
            else:
                if label == 1:
                    self.fp += 1
                else:
                    self.tn += 1

    def cal_f1(self):
        if 2 * self.tp + self.fp + self.fn == 0:   #注意 除0保护
            return 0
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)





if __name__ == "__main__":
    nodes = [[]]  # 标号0-n-1 -> 1-n
    n, m, k = map(int, input().split())
    for _ in range(n):
        l, r, f, th, label = map(int, input().split())
        nodes.append(Node(l, r, f, th, label))

    test_data = []  # (feature,label)
    for _ in range(m):
        sample = list(map(int, input().split()))
        test_data.append((sample[:-1], sample[-1]))


    def dfs(idx, data):
        # 叶节点
        ans = Result(0,0,0,0)
        ans.cal(data, nodes[idx].label)
        if nodes[idx].l == 0 and nodes[idx].r == 0:
            return ans
        # 非叶节点  考虑是否截断
        cur_f1 = ans.cal_f1()

        data1 = [x for x in data if x[0][nodes[idx].f - 1] <= nodes[idx].th]
        data2 = [x for x in data if x[0][nodes[idx].f - 1] > nodes[idx].th]
        ans1 = dfs(nodes[idx].l, data1)
        ans2 = dfs(nodes[idx].r, data2)
        ans3 = ans1 + ans2
        f1 = ans3.cal_f1()

        if cur_f1 > f1:  # 截断
            return ans
        return ans3  # 不截断


    ans = dfs(1, test_data)
    print(f"{ans.cal_f1():.6f}")
```

## 0828 留学生
### 1. 决策树test部分


```python
f, m, n = map(int, input().split())


class Node:
    def __init__(self, fi, th, l, r, label):
        self.fi = fi
        self.th = th
        self.l = l
        self.r = r
        self.label = label


nodes = []
for _ in range(m):
    fi, th, l, r, label = map(float, input().split())
    nodes.append(Node(int(fi), th, int(l), int(r), int(label)))
data = []
for i in range(n):
    sample = list(map(float, input().split()))
    data.append((sample, i))
res = [0] * n


def dfs(node_id, cur_data):
    if not cur_data:
        return
    # 叶节点
    root = nodes[node_id]
    if root.l == -1 and root.r == -1:
        for x in cur_data:
            res[x[1]] = root.label
    else:
        # 切割
        data_l = [x for x in cur_data if x[0][root.fi] <= root.th]
        data_r = [x for x in cur_data if x[0][root.fi] > root.th]
        dfs(root.l, data_l)
        dfs(root.r, data_r)


dfs(0, data)
for x in res:
    print(x)
```

### 2. Group卷积 
常规卷积：第二层循环直接`for out_ c in range(out_channels)`
Group卷积：将 X 和 K 按组切片，再进行卷积计算


```python
import sys
import numpy as np
def main():
    in_data = list(map(int, sys.stdin.readline().split()))
    batch_size, in_channels, height, width = map(int, input().split())
    kernel_data = list(map(int, sys.stdin.readline().split()))
    out_channels, k_channels, kernel_h, kernel_w = map(int, input().split())
    groups = int(input())
    # 处理error
    if (
        in_channels % groups != 0
        or out_channels % groups != 0
        or k_channels != in_channels // groups
    ):
        print(-1)
        print(-1)
        # return

    X = np.array(in_data).reshape((batch_size, in_channels, height, width))
    K = np.array(kernel_data).reshape((out_channels, k_channels, kernel_h, kernel_w))
    # 计算输出尺寸并初始化
    Out_h = height - kernel_h + 1
    Out_w = width - kernel_w + 1
    Output = np.zeros((batch_size, out_channels, Out_h, Out_w), dtype=np.int64)


    for b in range(batch_size):
        for g_idx in range(groups):
            o_start = g_idx * (out_channels // groups)
            o_end = o_start + out_channels // groups
            for i in range(o_start, o_end):
                for h in range(Out_h):
                    for w in range(Out_w):
                        # 剪切
                        roi = X[
                            b,
                            g_idx * k_channels : (g_idx + 1) * k_channels,
                            h : h + kernel_h,
                            w : w + kernel_w,
                        ]
                        kernel = K[i, :, :, :]
                        out = np.sum(roi * kernel)
                        Output[b, i, h, w] = out
    # 输出Output(4维展开)
    print(" ".join(map(str, Output.flatten().tolist())))
    # 输出尺寸
    print(batch_size, out_channels, Out_h, Out_w)

if __name__ == "__main__":
    main()
```

## 0903
### 第2题-大模型训练MOE场景路由优化算法
题目说明: n专家，均分到m个NPU，取p个NPU组(按照每个组max(专家概率))，p个组的专家中选k个专家输出。


```python
def main():
    n, m, p, k = map(int, input().split())
    person = list(map(float, input().split()))

    # 判断error
    if n % m != 0 or (n / m) * p < k:
        print("error")
        return

    groups = []  # (p1,...pr)一组
    r = n // m
    for i in range(m):
        groups.append(person[i * r : (i + 1) * r])

    groups_l = [(max(x), i) for i, x in enumerate(groups)]
    groups_l.sort(reverse=True)
    candidate = []  # 前p个组的专家  (pi,i)
    for i in range(p):
        idx = groups_l[i][1]
        for j in range(idx * r, (idx + 1) * r):  # 专家编号
            candidate.append((person[j], j))

    candidate.sort(reverse=True)  # 专家内部排序
    ans = [candidate[i][1] for i in range(k)]  # 选取前k个专家的编号
    ans.sort()  # 恢复升序
    print(" ".join(map(str, ans)))


if __name__ == "__main__":
    main()
```

### 第三题：云存储设备故障预测
题目说明：数据清洗、逻辑回归、预测输出

#### numpy 
##### 1. 基本操作


```python
"""
numpy基本操作:slice,数据筛选,axis,归一化
"""
import numpy as np
raw_list = [[1, 2], [3, 4]]
data = np.array(raw_list)
print(data.shape)  # (n,m)  n行m列

# slice [行，列]  和 python 保持一致
feature = data[:, :-1]
label = data[:, -1]

# 数据清洗、筛选
mask = data[:, 0] > 3  # 生成掩码[False,True,True]
clean_data = data[mask]  # 只保留第2、3行
# 示例
f, th = 1, 8.12
data_left = data[data[:, f] <= th]

# axis(以该维度形式输出)
means0 = np.mean(data, axis=0)  # 计算每列均值
means1 = np.mean(data, axis=1)  # 计算每行均值
stds = np.std(data, axis=0)  # 计算每列标准差

# 归一化(Z-score)
X_normalized = (data - means0)/(stds + 1e-8) # 防止除0 
probs = [0.1,0.3,0.7]
pred_class = np.argmax(probs) #返回最大值的下标
```

##### 2. 清洗数据


```python
import numpy as np
raw_list = []
data = np.array(raw_list)
# 1. 补充缺失值 data:(m * n)
for col_idx in range(data.shape(1) - 1):  # 除去label列
    col = data[:, col_idx]
    nan_mask = np.isnan(col)

    if np.any(nan_mask):  # 判断是否存在NaN
        col_mean = np.mean(col[~nan_mask])
        col[nan_mask] = col_mean

# 丢弃含有nan的行
# 1. np.isnan(data) -> 找 NaN
# 2. .any(axis=1) -> 这一行只要有一个 NaN 就返回 True
# 3. ~ -> 取反，即"这一行纯净"为 True
clean_data = data[~np.isnan(data).any(axis=1)]


# 2. 处理异常值
means = np.mean(data, axis=0)
std = np.std(data, axis=0)

# 设置阈值
threshold = 3
up_limit = means + 3 * std
low_limit = means - 3 * std

valid_mask = ((data < up_limit) & (data > low_limit)).all(axis=1)
clean_data1 = data[valid_mask]
```

#### 题解


```python
import numpy as np
# 处理输入
n = int(input())
raw_train_data = []
for _ in range(n):
    line = list(map(str, input().split(",")))
    raw_train_data.append(line)
m = int(input())
raw_test_data = []
for _ in range(m):
    line = list(map(str, input().split(",")))
    raw_test_data.append(line)

# 数据清洗
train_data = np.array(raw_train_data)
test_data = np.array(raw_test_data)


for col_idx in range(1, 6):
    col = train_data[:, col_idx]
    # 处理缺失值NaN
    valid_values = col[col != "NaN"].astype(float)
    # 处理异常值
    if col_idx in [1, 2]:
        valid_values = valid_values[valid_values >= 0]
    elif col_idx in [3, 4]:
        valid_values = valid_values[(valid_values >= 0) & (valid_values <= 1000)]
    elif col_idx == 5:
        valid_values = valid_values[(valid_values >= 0) & (valid_values <= 20)]
    means = np.mean(valid_values)
    medians = np.median(valid_values)
    # col填充缺失值
    for i in range(len(col)):
        if col[i] == "NaN":
            col[i] = means
        else:
            val = float(col[i])
            if col_idx in [1, 2] and val < 0:
                col[i] = medians
            elif col_idx in [3, 4] and (val < 0 or val > 1000):
                col[i] = medians
            elif col_idx == 5 and (val < 0 or val > 20):
                col[i] = medians
    col1 = test_data[:, col_idx]
    for i in range(len(col1)):
        if col1[i] == "NaN":
            col1[i] = means
        else:
            val = float(col1[i])
            if col_idx in [1, 2] and val < 0:
                col1[i] = medians
            elif col_idx in [3, 4] and (val < 0 or val > 1000):
                col1[i] = medians
            elif col_idx == 5 and (val < 0 or val > 20):
                col1[i] = medians
    train_data[:, col_idx] = col
    test_data[:, col_idx] = col1

# 2. 逻辑回归
X_train = train_data[:, 1:6].astype(float)
y_train = train_data[:, -1].astype(int)
m, n_features = X_train.shape

# 训练模型
alpha = 0.01
num_iterations = 100
weights, b = np.zeros(n_features), 0.0
for _ in range(num_iterations):
    linear_model = np.dot(X_train, weights) + b
    y_predicted = 1 / (1 + np.exp(-linear_model))
    # 计算梯度
    gradient = np.dot(X_train.T, (y_predicted - y_train)) / m
    db = np.sum(y_predicted - y_train) / m

    # 更新权重
    weights -= alpha * gradient
    b -= alpha * db

# 3. 预测输出
X_test = test_data[:, 1:6].astype(float)
final_z = np.dot(X_test, weights) + b
y_test_predicted = 1 / (1 + np.exp(-final_z))
y_test_labels = (y_test_predicted >= 0.5).astype(int)

for i in range(len(y_test_labels)):
    print(y_test_labels[i])
```

## 0904 留学生
### 第2题-大模型训练数据均衡分配算法
题目说明: 将m个数据，尽量均匀分配到n个计算节点上 即`min(l_max)`

#### 堆操作
```python
import heapq

# 定义堆/将list转为堆
min_heap = []
data = [5, 3, 8, 1, 2]
heapq.heapify(data)

# 增、删、查
heapq.heappush(min_heap, 4)  # 插入元素4
top = heapq.heappop(min_heap)  # 弹出最小元素
if min_heap:
	smallest = min_heap[0]  # 查看最小元素
res = heapq.heappushpop(min_heap, 6)  # 插入6并弹出最小元素

# 最大堆 （存入取反，取出再取反）
max_heap = []
val = 4
heapq.heappush(max_heap, -val)  # 插入元素-4
largest = -heapq.heappop(max_heap)  # 弹出最大元素

#获取前n个元素
top_n = heapq.nlargest(3, data)  # 获取前三个最大元素
bottom_n = heapq.nsmallest(2, data)  # 获取前两个最小元素
```

#### 题解


```python
# 思路: 最小堆(维护每组的负载) + 贪心(将当前数据分配给负载最小的组)
# 注意：数据应该从大到小入组
import sys
import heapq
def main():
    n = int(input())
    m = int(input())
    data = list(map(int, sys.stdin.readline().split()))
    data.sort(reverse=True)
    group_sum = [0] * n
    heapq.heapify(group_sum)
    ans = 0
    for x in data:
        cur = heapq.heappop(group_sum)
        cur += x
        ans = max(ans, cur)
        heapq.heappush(group_sum, cur)
    print(ans)

if __name__ == "__main__":
    main()
```

### 第3题 神经网络(softmax + Fc + softmax + Fc)
题目说明: 实现一个两层神经网络的前向传播，包含两个softmax层和两个全连接层。

## 0905 塔子哥模拟
### 第二题-阈值最优的决策树
题目说明：对于feature_n = 1的情况，枚举阈值，计算最优准确率


```python
# 由于特征是1维的，可以排序+前后缀直接计算准确率(不需要dfs、th具体的值了)
class Node:
    def __init__(self, th, l, r, label=-1):
        self.th = th
        self.l = l
        self.r = r
        self.label = label


M = int(input())
data = []  # [(feature,label), ...]
th = set()  # [f1,f2,...]
nodes = [Node(0, 1, 2)]  # 初始化根节点

for _ in range(M):
    f, l = map(int, input().split())
    data.append((f, l))
    th.add(f)

y_l, y_r = map(int, input().split())
nodes.append(Node(-1, -1, -1, y_l))
nodes.append(Node(-1, -1, -1, y_r))


def dfs(idx, data):  # 返回[正确个数，总个数]
    if not data:
        return [0, 0]

    root = nodes[idx]

    # 叶节点，统计、返回
    if root.l == -1 and root.l == -1:
        label = root.label
        cnt = sum(1 for x in data if x[1] == label)
        return [cnt, len(data)]
    else:
        data_l = [x for x in data if x[0] <= root.th]
        data_r = [x for x in data if x[0] > root.th]
        res1 = dfs(root.l, data_l)
        res2 = dfs(root.r, data_r)
        return [res1[0] + res2[0], res1[1] + res2[1]]


ans = 0
for t in th:
    nodes[0].th = t
    res = dfs(0, data)
    ans = max(ans, res[0] / res[1])
print(f"{ans:.3f}")
```


```python
import numpy as np
# 数据处理
L, D = map(int, input().split(","))
raw_in_seq = list(map(float, input().split(",")))
raw_W_q1 = list(map(float, input().split(",")))
raw_W_k1 = list(map(float, input().split(",")))
raw_W_v1 = list(map(float, input().split(",")))
raw_W_fc1 = list(map(float, input().split(",")))
raw_b_fc1 = list(map(float, input().split(",")))
raw_W_q2 = list(map(float, input().split(",")))
raw_W_k2 = list(map(float, input().split(",")))
raw_W_v2 = list(map(float, input().split(",")))
raw_W_fc2 = list(map(float, input().split(",")))
raw_b_fc2 = list(map(float, input().split(",")))

in_seq = np.array(raw_in_seq).reshape(L, D)
W_q1 = np.array(raw_W_q1).reshape(D, D)
W_k1 = np.array(raw_W_k1).reshape(D, D)
W_v1 = np.array(raw_W_v1).reshape(D, D)
W_fc1 = np.array(raw_W_fc1).reshape(D, D)
b_fc1 = np.array(raw_b_fc1).reshape(D)
W_q2 = np.array(raw_W_q2).reshape(D, D)
W_k2 = np.array(raw_W_k2).reshape(D, D)
W_v2 = np.array(raw_W_v2).reshape(D, D)
W_fc2 = np.array(raw_W_fc2).reshape(D, D)
b_fc2 = np.array(raw_b_fc2).reshape(D)


def Attention(Q, K, V):
    z = np.dot(Q, K.T) / D**0.5
    y = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return np.dot(y, V)


Q1, K1, V1 = np.dot(in_seq, W_q1), np.dot(in_seq, W_k1), np.dot(in_seq, W_v1)
y1 = Attention(Q1, K1, V1)
z1 = np.dot(y1, W_fc1) + b_fc1
Q2, K2, V2 = np.dot(z1, W_q2), np.dot(z1, W_k2), np.dot(z1, W_v2)
y2 = Attention(Q2, K2, V2)
z2 = np.dot(y2, W_fc2) + b_fc2
final = z2.flatten().tolist()
for i in range(len(final)):
    if i == len(final) - 1:
        print(f"{final[i]:.2f}")
    else:
        print(f"{final[i]:.2f}", end=",")

```

### 第三题 随机游走概率
题目说明：从节点s开始，走k步，不能经过x的概率。
超时->`from collections import cache`!


```python
from functools import cache

@cache
def dfs(s, k):
    if k == 0:
        return 1
    else:
        res = 0
        for c in choices[s]:
            res += dfs(c, k - 1) * P[s][c]
        return res

if __name__ == "__main__":
    n = int(input())
    s, x, k = map(int, input().split())
    s -= 1 
    x -= 1
    P = []
    choices = []
    for _ in range(n):
        tmp = list(map(float, input().split()))
        tmp_choices = []
        for i in range(n):
            if tmp[i] != 0 and i != x:
                tmp_choices.append(i)
        choices.append(tmp_choices)
        P.append(tmp)
    ans = dfs(s, k)
    print(f"{ans:.6f}")
```

## 0910
太难了！
### 第一题 滑动窗口计算文本相似度


```python
# a 了30% 后面有时间再看吧！
doc_num = int(input())
doc_list = []  # [(word1:n1, word2:n2, ...)]
for _ in range(doc_num):
    curr_doc = Counter(map(str, sys.stdin.readline().split()))
    doc_list.append(curr_doc)
k = int(input())
query_times = int(input())
queries_list = []  # (t,词list)
for _ in range(query_times):
    line = list(map(str, sys.stdin.readline().split()))
    t, one_query = int(line[0]), line[1:]
    queries_list.append((t, one_query))

# 计算TF(wi,q)
TF_query = Counter()  # wi:频率
for x in queries_list:
    each_query = x[1]
    TF_query += Counter(each_query)


def calculate_IDF(one_query, win_docs):
    """
    计算单个query中每个单词在win_docs中的IDF值，返回dict
    win_docs : 每个文章的counter
    """
    IDF_list = dict()
    N = len(win_docs)
    for w in one_query:
        Nx = 0
        if w not in IDF_list:
            for each_doc in win_docs:
                if w in each_doc:
                    Nx += 1
            IDF_list[w] = math.log((N + 1) / (Nx + 1)) + 1
    return IDF_list


def CosineSimilarity(vec1, vec2):
    """
    计算vec1、vec2余弦相似度,返回float
    """
    r1 = 0  # 内积
    s1 = s2 = 0  # 平方和
    for k1, k2 in zip(vec1, vec2):
        r1 += k1 * k2
        s1 += k1**2
        s2 += k2**2
    if s1 == 0 or s2 == 0:
        return 0
    else:
        return r1 / (s1 * s2) ** 0.5


final_ans = []
for q in queries_list:
    ans = [0, 0]
    win_end, one_query = q[0], q[1]
    # 计算窗口 start - end
    win_start = max(win_end - k + 1, 0)
    # TF(wi,doc)
    win_doc = doc_list[win_start : win_end + 1]
    # IDF(wi)
    IDF = calculate_IDF(one_query, win_doc)
    # vec_q
    len_TF_query = sum(x for x in TF_query.values())
    vec_q = [TF_query[w] / len_TF_query * IDF[w] for w in one_query]
    for idx in range(win_start, win_end + 1):
        weight = (idx - win_end + k) / k
        TF_doc = doc_list[idx]
        vec_d = []
        for w in one_query:
            if w in TF_doc:
                vec_d.append(TF_doc[w] / len(TF_doc) * IDF[w] * weight)
            else:
                vec_d.append(0)
        curr_ans = CosineSimilarity(vec_d, vec_q)
        if curr_ans >= 0.6 - 1e-12:
            if curr_ans > ans[0]:
                ans = [curr_ans, idx]
    if ans[0] == 0:
        final_ans.append(-1)
    else:
        final_ans.append(ans[1])
print(" ".join(map(str, final_ans)))
```

## 0912
### 第2题-二叉树中序遍历的第k个祖先节点
层序遍历建立树,dfs中序遍历树(**尽量少copy，直接在变量上修改**)，deque的使用


```python
class Node:
    def __init__(self, val, l=-1, r=-1, path=[]):
        self.val = val
        self.l = l
        self.r = r


level_search = list(map(str, sys.stdin.readline().split()))
nodes_num = len(level_search)
u, k = map(int, input().split())
q = deque()
nodes = []


def dfs(cur_path, idr, u, k):
    """
    cur_path: 中序遍历的ancestor
    """
    root = nodes[idr]
    if not root:
        return -1
    if root.val == u:
        if len(cur_path) < k:
            return -1
        else:
            return cur_path[-k]
    else:
        if root.l != -1:
            res = dfs(cur_path, root.l, u, k)
            if res != -1:
                return res
        if root.r != -1:
            cur_path.append(root.val)
            res = dfs(cur_path, root.r, u, k)
            cur_path.pop()
            if res != -1:
                return res
    return -1


if not level_search or level_search[0] == "#":
    print(-1)
else:
    idx = 1  # 下一个要取的点
    root_val = int(level_search[0])
    root = Node(root_val)
    q.append(root)
    nodes.append(root)
    ans = -2
    # 构建树
    while q:
        curr_root = q[0]
        if idx < nodes_num:
            val = level_search[idx]
            if val != "#":
                left_root = Node(int(val))
                q.append(left_root)
                curr_root.l = len(nodes)
                nodes.append(left_root)
            idx += 1
        if idx < nodes_num:
            val = level_search[idx]
            if val != "#":
                right_root = Node(int(val))
                q.append(right_root)
                curr_root.r = len(nodes)
                nodes.append(right_root)
            idx += 1
        q.popleft()
    res = dfs([], 0, u, k)
    print(res)

```

## 0917
### 第二题：大模型attention开发
复现attention
**四舍五入`f"{ans:0f}"`**
**矩阵乘法@**
**转成上三角矩阵np.triu(),下三角矩阵np.tril()**


```python
n, m, h = map(int, input().split())
X = np.ones((n, m))
W1 = W2 = W3 = np.ones((m, h))
for i in range(1, m):
    for j in range(min(i, h)):
        W1[i][j] = 0
        W2[i][j] = 0
        W3[i][j] = 0
Q = np.dot(X, W1)
K = np.dot(X, W2)
V = np.dot(X, W3)
M = np.dot(Q, K.T) / (h**0.5)
softmax_M = M / np.sum(M, axis=1, keepdims=True)
y = np.dot(softmax_M, V)
ans = int(np.sum(y))
print(np.rint(ans))
```

### 第三题：大模型分词
给定一个文本text、划分的概率，找到最大的分词收益
**看下数据的范围，有时失败标志设置不合理**

**python中浮点数最大值、最小值:float('inf')、float('-inf')**
**整数的最大值、最小值可以自己写一个很大的,`INT_MAX = 10**12 INT_MIN = - 10 ** 12`**
**np中有`np.inf`**


```python
@cache
def dfs(content, last):
    """
    :param t: 剩下的文本
    :param last: 上一个选择的词
    return:max(sum p,x)
    """
    if not content:
        return 0
    ans = -100000
    for i in range(1, len(content) + 1):  # 枚举切割的点
        word = content[:i]
        if word in word_dict:
            sub_dfs = dfs(content[i:], word)
            if sub_dfs != -100000:
                if (
                    last in influence_matrix_content
                    and word in influence_matrix_content[last]
                ):
                    for x in influence_matrix_prob[last]:
                        if x[0] == next:
                            sub_dfs += x[1]
                            break
                ans = max(ans, sub_dfs + word_dict[word])
    return ans


text = input()
n = int(input())
word_dict = {}
for _ in range(n):
    word, p = map(str, input().split())
    word_dict[word] = float(p)
m = int(input())
influence_matrix_prob = defaultdict(list)  # [w] : [(w1:p1),(w1:p2)]
influence_matrix_content = defaultdict(list)  # [w] : [w1,w2,...]
for _ in range(m):
    start, next, prob = map(str, input().split())
    influence_matrix_content[start].append(next)
    influence_matrix_prob[start].append((next, float(prob)))
ans = dfs(text, "")
if ans == -100000:
    print(0)
else:
    print(ans)

```

## 0918(留学生)
### 第二题：最大能量路径
卷积 + dp
**以后多维数组直接np.array(())**
**print(1) exit(0)可以帮忙排除错误的位置**
**数组越界还有可能是初始化错了**


```python
line = list(map(int, input().split()))
h, w, k = line[0], line[1], line[2]
raw_inputs = []
for _ in range(h):
    cur_line = list(map(float, input().split()))
    raw_inputs.append(cur_line)
raw_kernel = []
for _ in range(k):
    cur_line = list(map(float, input().split()))
    raw_kernel.append(cur_line)
inputs = np.array(raw_inputs)
kernel = np.array(raw_kernel)


def Conv2d(input1, Kernel, padding=0):
    h_in, w_in = input1.shape
    kh, kw = Kernel.shape

    h_out = h_in + 2 * padding - kh + 1
    w_out = w_in + 2 * padding - kw + 1
    # 填充padding
    if padding > 0:
        input1 = np.pad(
            input1, ((padding, padding), (padding, padding)), mode="constant"
        )

    Output = np.zeros((h_out, w_out))
    for h in range(h_out):
        for w in range(w_out):
            roi = input1[h : h + kh, w : w + kw]
            cur_val = np.sum(roi * Kernel)
            Output[h, w] = cur_val
    return Output


Energy = Conv2d(inputs, kernel, k // 2)  # h * w
dp = np.array((h,w))
for col in range(w):
    for row in range(h):
        if col == 0:  # 第一列
            dp[row, col] = Energy[row, col]
        else:
            add1 = dp[row, col - 1]
            if row - 1 >= 0:
                add1 = max(add1, dp[row - 1, col - 1])
            if row + 1 < h:
                add1 = max(add1, dp[row + 1, col - 1])
            dp[row, col] = Energy[row, col] + add1
ans = 0
for row in range(h):
    ans = max(dp[row][-1], ans)
print(f"{ans:.1f}")
```

### 第三题 数据中心水温调节档位决策
实现一个多分类器，sgd，softmax
**1. 一般设置`lr = 0.03`,`epochs = 500`**
**2. 梯度衰减：`if epoch % 150 == 0: lr *= 0.9`**
**3. 数据归一化(防止梯度爆炸NAN)**
```python
means = np.mean(X,axis = 0)
std = np.std(X,axis = 0)
X = (X - means) / std
test_x = (test_x - means) / std
```


```python
# 题解
line = list(map(int, sys.stdin.readline().split()))
feature_num, category_num, test_num = line[0], line[1], line[-1]
each_category_num = line[2:-1]
raw_train_data = []  # [s1,s2,...]
raw_train_y = []
for idx, xnum in enumerate(each_category_num):
    cur_label = [0] * category_num
    cur_label[idx] = 1
    for _ in range(xnum):
        sample = list(map(float, sys.stdin.readline().split()))
        raw_train_data.append(sample)
        raw_train_y.append(cur_label)
raw_test_data = []
for _ in range(test_num):
    sample = list(map(float, sys.stdin.readline().split()))
    raw_test_data.append(sample)


X = np.array(raw_train_data)
Y = np.array(raw_train_y)
test_x = np.array(raw_test_data)
W = np.zeros((feature_num, category_num))
b = np.zeros((1, category_num))

# 数据归一化
means = np.mean(X, axis=0)
std = np.std(X, axis=0) + 1e-8  # 防止除以0
X = (X - means) / std
test_x = (test_x - means) / std
lr = 0.03
epochs = 600


def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)


for epoch in range(epochs):
    z = X @ W + b
    P = softmax(z)

    dw = X.T @ (P - Y) / X.shape[0]
    db = np.mean((P - Y), axis=0, keepdims=True)

    W -= lr * dw
    b -= lr * db

# 测试
test_y = test_x @ W + b
# 输出每一行最大数所在的id
for i in range(test_y.shape[0]):
    print(np.argmax(test_y[i]))

```

## 0924
### 第三题 基于决策树的无线状态预策

1. **y初始化无脑`np.array(n)`**:`((n,1))`和`(n)`,前者是二维数组，后者是一维数组
2. **划分时，在计算信息熵/gini系数前，先排除`len(si) == 0`也就是没划分成功的情况**
3. **在选取合适的f_id时，直接用平常理解的max信息增益，max gini系数就好，别绕来绕去**
4. **predict就一个个样本预测，别想着批量预测(维护样本id很麻烦)**
5. `y.flatten()`把一个np.array从多维变一维
6. entropy计算时，注意log2(0)的情况


```python
n, m = map(int, input().split())
train_x = np.zeros((n, m))
train_y = np.zeros(n, dtype=int)
for i in range(n):
    line = list(map(int, sys.stdin.readline().split()))
    train_x[i] = line[:-1]
    train_y[i] = line[-1]
q = int(input())
test_x = np.zeros((q, m))
for i in range(q):
    line = list(map(int, sys.stdin.readline().split()))
    test_x[i] = line


class Node:
    def __init__(self, fid=None, l=None, r=None, label=None):
        self.fid = fid
        self.l = l
        self.r = r
        self.label = label


def cal_entropy(y):
    n_sample = y.shape[0]
    p0 = np.sum(y == 0) / n_sample
    p1 = np.sum(y == 1) / n_sample
    return -p0 * math.log2(p0 + 1e-12) - p1 * math.log2(p1 + 1e-12)


def build_tree(x, y):
    n_samples, n_features = x.shape
    # 所有样本label相同
    if len(np.unique(y)) == 1:
        return Node(label=y[0])

    max_gain = 0
    entr0 = cal_entropy(y)
    max_split_res = None
    for f_id in range(n_features):
        # 切分
        s0 = y[x[:, f_id] == 0]
        s1 = y[x[:, f_id] == 1]
        if len(s0) == 0 or len(s1) == 0:
            continue
        cur_split_val = (
            len(s0) * cal_entropy(s0) + len(s1) * cal_entropy(s1)
        ) / n_samples
        gain = entr0 - cur_split_val
        if gain > max_gain and gain > 1e-9:
            max_gain = gain
            max_split_res = [f_id, s0, s1, x[x[:, f_id] == 0], x[x[:, f_id] == 1]]

    # 没有feature可划分
    if not max_split_res:
        y0, y1 = np.sum(y == 0), np.sum(y == 1)
        if y0 >= y1:
            return Node(label=0)
        else:
            return Node(label=1)

    # 递归
    lnode = build_tree(max_split_res[3], max_split_res[1])
    rnode = build_tree(max_split_res[4], max_split_res[2])
    return Node(fid=max_split_res[0], l=lnode, r=rnode)


root = build_tree(train_x, train_y)
for sample in test_x:
    pr = root
    while True:
        # 叶
        if pr.l == None and pr.r == None:
            print(pr.label)
            break
        if sample[pr.fid] == 0:
            pr = pr.l
        else:
            pr = pr.r
```

## 0928
### 第2题-Yolo检测器中的anchor聚类
**kmeans。注意题目中说`聚类的度量为anchor`，意思是分类、计算是否需要停止时，都用anchor距离**
多读题！


```python
import numpy as np

N, K, T = map(int, input().split())
raw_x = []
for _ in range(N):
    cur_x = list(map(int, input().split()))
    raw_x.append(cur_x)
X = np.array(raw_x)


def kmeans(x, k, max_iter=1000, tol=1e-4):
    n_sample = x.shape[0]
    # 初始化簇中心
    centroid = x[0:k]

    for _ in range(max_iter):
        # E:计算dist，划分新的簇
        distance = np.zeros((n_sample, k))
        for idx in range(n_sample):
            for idc in range(k):
                distance[idx, idc] = 1 - cal_iou(x[idx], centroid[idc])
        labels = np.argmin(distance, axis=1)

        # M:更新簇中心
        new_centroid = np.zeros_like(centroid)
        for i in range(k):
            cluster = x[labels == i]
            if cluster.shape[0] == 0:
                new_centroid[i] = centroid[i]
            else:
                new_centroid[i] = np.mean(cluster, axis=0) // 1

        # 计算是否需要停止
        sum_b = 0
        for p1, p2 in zip(new_centroid, centroid):
            sum_b += 1 - cal_iou(p1, p2)
        
        if sum_b < tol:
            break
        centroid = new_centroid
    return centroid


def cal_iou(b1, b2):
    w1, h1 = b1
    w2, h2 = b2
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-16)


center_points = kmeans(X, K, T)
final_centers = center_points.tolist()
final_centers.sort(key=lambda x: x[0] * x[1], reverse=True)
for p in final_centers:
    print(p[0], p[1])
```

## 1010(留学生ai)
### 第3题-基于逻辑回归的意图分类器
- **出现batch_size:** `for i range(0,n_sample,batch_size)`，在计算`dw`和`db`时，分母是`cur_batch_len = X_batch.shape[0]`
- **ch的ascii值ord**
- **numpy的切片问题：**`x_batch = x[i]`时，第0维度消失，即`(features_n,)`。用`x_batch = x[i:i + 1]`可以保持维度`(1,feature_num)`


```python
N, M = map(int, input().split())
raw_train_x = [[0 for _ in range(7)] for _ in range(N)]
raw_train_y = []
raw_test_x = [[0 for _ in range(7)] for _ in range(M)]


for i in range(N):
    cur_feature, cur_label = map(str, input().split())
    raw_train_y.append(int(cur_label))
    cur_feature_cnt = Counter(cur_feature)
    for k in cur_feature_cnt.keys():
        idch = ord(k) - ord('A')
        raw_train_x[i][idch] = 1
for i in range(M):
    cur_feature = input()
    cur_feature_cnt = Counter(cur_feature)
    for k in cur_feature_cnt.keys():
        idch = ord(k) - ord('A')
        raw_test_x[i][idch] = 1

train_x, test_x, y = np.array(raw_train_x), np.array(raw_test_x), np.array(raw_train_y)
W, b = np.zeros((7, 1)), 0
epochs = 20
lr = 0.1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for _ in range(epochs):
    for i in range(N):  # batch_size = 1
        cur_x, y_true = train_x[i, :], y[i]
        cur_y = cur_x @ W + b
        cur_a = sigmoid(cur_y)

        dz = cur_a - y_true
        dw = cur_x.T * dz
        db = dz

        W -= lr * dw
        b -= lr * db

z_pre = test_x @ W + b
a_pre = sigmoid(z_pre)
for v in a_pre:
    if v > 0.5:
        print(1)
    else:
        print(0)

```

## 1010
### 第2题 数据聚类以及噪声点识别
1. 题目：DBSCAN(eps领域、核心点、直接密度可达、密度可达、密度相连) 本质：核心点吸收领域内node

2. 使用：
	- **Union-Find**:`parents`,`union`,`find`
	- **广播**：`N - 1`直接减；`N,D - k,D` -> `N,1,D - 1,k,D`得到`N,k`
	- **真值矩阵的sum：**True和False可以视为1/0
	- **np.where()**：返回的是一个`tuple`,分别是每个点的第0维list，第1维list（**按照点来的，可以重复！**）
	- `a = list(range(n))`


```python
import numpy as np
from collections import defaultdict

line = list(map(str, input().split()))
eps, min_sample, n_sample = float(line[0]), int(line[1]), int(line[2])
raw_data = []
for _ in range(n_sample):
    cur_sample = list(map(float, input().split()))
    raw_data.append(cur_sample)
points = np.array(raw_data)


# 计算核心点
dist_matrix = np.sqrt(
    np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2)
)
# sum [True/False]矩阵
neighbor_count = np.sum(dist_matrix < eps, axis=1)
core_pid = np.where(neighbor_count > min_sample)[0]

# [init range list]
parents = list(range(n_sample))  # 维护每个点的root


def union(x, y):
    rootX, rootY = find(x), find(y)
    if rootX != rootY:
        parents[rootX] = rootY  # 将X合并到Y上


def find(x):
    if parents[x] != parents[parents[x]]:
        parents[x] = find(parents[x])
    return parents[x]


# 找簇、噪声
noise_id = []
core_points_data = points[core_pid]

# [广播]
diff = points[:, np.newaxis, :] - core_points_data[np.newaxis, :, :]
dists = np.sum(diff**2, axis=2)
for j in range(len(core_pid)):
    # [where] 返回的是tuple (array([]),array([]))即每维满足条件的index
    neighbors_id = np.where(dists[:, j] < eps**2)[0]
    for nid in neighbors_id:
        union(core_pid[j], nid)

# 确定簇的个数
root_map = defaultdict(list)
for i in range(n_sample):
    rooti = find(i)
    root_map[rooti].append(i)
valid_cluster = 0
noise_cnt = 0
for k, v in root_map.items():
    if len(v) == 1:
        noise_cnt += 1
    else:
        valid_cluster += 1
print(valid_cluster, noise_cnt)

```

## 1015
### 第三题：基于二分Kmeans算法的子网分割问题
**认真读题！(初始化聚类中心是看x_min和x_max去选点，而不是f上所有的min和所有的max作为聚类中心)**


```python
import numpy as np


def kmeans_2(x, k=2, max_iter=1000, tol=1e-6):
    n_samples = x.shape[0]
    # 初始化簇心
    center = np.zeros((2, 2))
    min_x_idx = np.argmin(x[:,0])
    max_x_idx = np.argmax(x[:,0])
    center[0, :] = x[min_x_idx].copy()
    center[1, :] = x[max_x_idx].copy()

    for _ in range(max_iter):
        # E:计算距离，确定label
        # (N,1,D) - (1,K,D)
        diff = x[:,np.newaxis,:] - center[np.newaxis,:,:]
        distance = np.sum(diff**2,axis = 2 )
        labels = np.argmin(distance, axis=1)
        
		# M:更新簇中心
        new_center = np.zeros_like(center)
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                new_center[i] = np.mean(x[mask],axis = 0)
            else:
                new_center[i] = center[i]
        # 判断是否需要终止
        diff1 = new_center - center
        center = new_center
        if np.sum(np.sqrt(np.sum(diff1**2, axis=1))) < tol:
            break
    return [x[labels == 0], x[labels == 1]]


def cal_see(x):
    center = np.mean(x, axis=0)
    diff = x - center
    return np.sum(diff**2)


N = int(input())
M = int(input())
raw_x = []
for _ in range(M):
    sample = list(map(int, input().split()))
    raw_x.append(sample)
x = np.array(raw_x)
clusters = []
clusters.append(x)
for _ in range(N - 1):
    min_see_id, min_see_val = None, float("-inf")
    min_divide = []
    for id_clu, cur_cluster in enumerate(clusters):
        see0 = cal_see(cur_cluster)
        clu0, clu1 = kmeans_2(cur_cluster)
        if clu0.shape[0] == 0 or clu1.shape[0] == 0:
            continue
        see1 = cal_see(clu0) + cal_see(clu1)
        if see0 - see1 > min_see_val:
            min_see_id = id_clu
            min_see_val = see0 - see1
            min_divide = [clu0, clu1]
    # 选择最小的进行更新
    clusters[min_see_id] = min_divide[0]
    clusters.append(min_divide[1])
    # 输出
    clusters.sort(key=lambda x: len(x), reverse=True)
    res = [str(i.shape[0]) for i in clusters]
    print(" ".join(res))
```

## 1106(留学生ai)
### 第2题-医疗诊断模型的训练与更新
- 题目说明：复现一个神经网络，输出y_pre，计算mse，输出更新后的W
- 总结：
    1. 梯度计算:乘法项`C = AB`**看形状dC、A拼成dB**，偏置`C = AB + b`**dC累和，消去一个维度**，逐元素`A = X*Y`或者激活函数`sigmoid`和`relu`**dZ = dA\*Z的导数**
    2. **numpy输出**：
        - 同行一个个输出（,隔开）：`print(*(f"{x:.2f}" for x in arr.flatten()),sep = ",")` 逐行输出 `sep="\n"`
    3. 扩充`np.tile(x,(L,1))`将x在axis = 0 复制L次，axis = 1复制1次（也就是直接copy）
        


```python
L, D, K, lr = map(float, input().split(","))
L, D, K = int(L), int(D), int(K)
y_true = np.zeros(K)
y_true = list(map(float, input().split(",")))

x = np.zeros((L, D))
line = list(map(float, input().split(",")))
for i in range(L):
    x[i] = line[i * D : i * D + D]

W_mlp = np.zeros((D, D))
line = list(map(float, input().split(",")))
for i in range(D):
    W_mlp[i] = line[i * D : i * D + D]

W_cls = np.zeros((D, K))
line = list(map(float, input().split(",")))
for i in range(D):
    W_cls[i] = line[i * K : i * K + K]

z1 = x @ W_mlp
y_pre = z1 @ W_cls

# 输出平均概率
mean_y = y_pre.mean(axis=0)
print(*(f"{x:.2f}" for x in mean_y))

# 输出MSE
mse = np.mean((mean_y - y_true) ** 2)
print(f"{mse:.2f}")

dy_mean = 2 * (mean_y - y_true) / K
dy = np.tile(dy_mean, (L, 1)) / L
# 输出 W_mlp W_cls
dw_cls = z1.T @ dy
dz1 = dy @ W_cls.T
dw_mlp = x.T @ dz1
W_cls -= lr * dw_cls
W_mlp -= lr * dw_mlp
print(*(f"{x:.2f}" for x in W_mlp.flatten()))
print(*(f"{x:.2f}" for x in W_cls.flatten()))
```

## 1112
### 第二题:基于全连接层的INT8非对称量化实现
1. masked的应用:` v[c1_mask] = v[c1_mask] // 1`
2. 打印np.array:`print(*y_q.astype(int))` ,`*`表示将数组拆开一个个传给print,默认用空格隔开
3. 初始化非0/1的array:`np.full((m,n),-128)`
4. `np.amin()`和`np.amax()`:返回array的最小/大值,可以用axis进行筛选
5. **`np`方法和python自带的方法不能混用!**
    - `np.sum(a)`: 默认 flatten 后求和，返回一个标量。
    - `sum(a)`: Python 自带的 sum 会遍历第一维，把它当成 list of rows，然后把两行向量相加 [1,2] + [3,4]。它返回的是一个向量而不是数字！
    -  `np.max(a)`: 输出 4 (全图最大值)。
    - `max(a)`: Python 自带的 max 会比较第一维的元素（也就是比较 [1,2] 和 [3,4] 这两个行向量）。根据 Python 规则，它比较第一个元素 3>1，所以判断第二行更大，返回了整行。
    -  **可以直接调用对象的方法，这样不会出错：** `x.sum()`等价于`np.sum(x)` ,`x.max()`,`x.mean()`


```python
import sys
from collections import defaultdict, Counter
import numpy as np

n = int(input())
x = np.zeros(n)
X = list(map(float, sys.stdin.readline().split()))
m, n1 = map(int, input().split())
W = np.zeros((m, n))
for i in range(m):
    line = list(map(float, sys.stdin.readline().split()))
    W[i] = line


def cal_scale(v):
    max_v, min_v = np.amax(v), np.amin(v)
    return (max_v - min_v) / 255


def round(v):
    v_l = v - v // 1
    c1_mask = v_l < 0.5
    c2_mask = v_l > 0.5
    c3_mask = v_l == 0.5
    v[c1_mask] = v[c1_mask] // 1
    v[c2_mask] = v[c2_mask] // 1 + 1
    v[c3_mask] = (v[c3_mask] + 1) // 2 * 2
    return v


def clamp(v, lo, hi):
    c1_mask = v < lo
    c2_mask = v > hi
    v[c1_mask] = lo
    v[c2_mask] = hi
    return v


# 非对称量化
if cal_scale(X) != 0:
    X_q = clamp(round((X - np.amin(X)) / cal_scale(X)) - 128, -128, 127)
else:
    X_q = np.array([-128 for _ in range(n)])
if cal_scale(W) != 0:
    W_q = clamp(round((W - np.amin(W)) / cal_scale(W)) - 128, -128, 127)
else:
    W_q = np.array([[-128 for _ in range(n)] for _ in range(m)])
Y_q = X_q @ W_q.T
print(*Y_q.astype(int))

# 反量化
X_dq = (X_q + 128) * cal_scale(X) + np.amin(X)
W_dq = (W_q + 128) * cal_scale(W) + np.amin(W)
Y_dq = X_dq @ W_dq.T
Y = X @ W.T
mse = np.mean((Y_dq - Y) ** 2)
print(f"{mse*100000:.0f}")
```

## 1120(留学生ai)
### 第2题-Vision Transformer中的Patch Embdding层实现
理解思路:
1. 输入图片`(H,W,C)`，划分成`(num_patches, patch_size, patch_size, C)`，其中`num_patches = (H/patch_size) * (W/patch_size)` 也就是题目中的`(N*N个,p_size,p_size,C)`
2. 使用`reshape()`将每个patch展平`(p_size*p_size*C,)`    
3. 全连接层映射到`embed_dim`维度  `(p_size*p_size*c,) * (p_size*p_size*c,embed_dim) -> (embed_dim,)`
4. 拼接CLS token  `(1,embed_dim) + (num_patches,embed_dim) -> (num_patches+1,embed_dim)`



```python
import numpy as np


def patch_embedding_numpy(images, patch_size=32, embedding_dim=384):
    """
    输入 images: (B, C, H, W) -> e.g. (1, 3, 448, 448)
    """
    B, C, H, W = images.shape

    grid_h, grid_w = H / patch_size, W / patch_size
    num_patches = grid_h * grid_w

    # 拆分:(B,C,H,W) -> (B,C,grid_h,patch_size,grid_w,patch_size)
    step1 = images.reshape(B, C, grid_h, patch_size, grid_w, patch_size)
    # 调整维度 (b,c,gh,ps,gw,ps) -> (b,gh,gw,c,ps,ps)
    step2 = step1.transpose(0, 2, 4, 1, 3, 5)
    # flatten
    patch_flat_dim = patch_size * patch_size * C
    step3 = step2.reshape(B, num_patches, patch_flat_dim)

    # 初始化系数矩阵E,b
    E = np.zeros((patch_flat_dim, embedding_dim))
    b = np.zeros(embedding_dim)

    step4 = step3 @ E + b

    # 加上CLS(可学习,初始化为0)
    cls_token = np.zeros((B, 1, embedding_dim))

    final_output = np.concatenate([cls_token, step4], axis=1)
    return final_output

```

## 1203
### 第2题 神经网络剪枝
- `np.argsort(array,axis)` 按照axis排序(升序)后返回对应的index
- `print(*(int(x) for x in labels.flatten()))` 换格式逐行输出
- `keepdims=True`在softmax函数中必须写


```python
import sys
from collections import defaultdict, Counter
import numpy as np


n, d, c = map(int, input().split())
X = np.zeros((n, d))
for i in range(n):
    line = list(map(float, input().split()))
    X[i] = line
W = np.zeros((d, c))
for i in range(d):
    line = list(map(float, input().split()))
    W[i] = line
ratio = float(input())
# 计算k
k = int(ratio * d // 1)
if ratio > 0 and k == 0:
    k = 1

# 计算L1，筛选保留的特征列
W_L1 = np.argsort(np.sum(np.abs(W), axis=1))
remain_lines_index = W_L1[k:]

# 裁剪
X1 = X[:, remain_lines_index]
W1 = W[remain_lines_index, :]
h = X1 @ W1


def softmax(x):
    exp_x = np.exp(x - np.amax(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


y = softmax(h)
labels = np.argmax(y, axis=1)
print(*(int(x) for x in labels.flatten()))
```

### 第3题 簇大小均匀的kmeans
1. 全整数直接`dtype = int`
2. array排序
    - `np.sort(x,axis= 0)`表示对每一列独立排序，**会破坏行的完整性**
    -  转为`list`排序。python 原生`.sort()`支持字典序。
        ```python
        points_list = centers.tolist()
        points_list.sort()
        sorted_centers = np.array(points_list)
        ```
    - 利用`np.argsort(x,axis=0)`记录排序好的index，然后用索引来排序
3. `array_equal(a,b)`判断矩阵a和b是否相等，相等返回true。
4. `argsort()`和`argmin()`和`argmax()`，返回的是index 


```python
import numpy as np
n, m, k = map(int, input().split())
clients = np.zeros((n, m), dtype=int)
for i in range(n):
    line = list(map(int, input().split()))
    clients[i] = line
new_client = np.zeros(m, dtype=int)
new_client = list(map(int, input().split()))


# kmeans
def kmeans(inputs, k, tol=0):
    n_samples, n_features = inputs.shape
    # 初始化簇中心
    centers = inputs[:k]
    # 计算每个簇的最大容量
    inter_min = n_samples // k
    cluster_inter_max = [inter_min] * k
    for i in range(n_samples - inter_min * k):
        cluster_inter_max[i] += 1
    labels = np.zeros(n_samples, dtype=int)
    while True:
        # E :计算距离
        new_labels = np.zeros(n_samples, dtype=int)
        cluster_inter_cur = np.zeros(k, dtype=int)
        for i in range(n_samples):
            cur_client = inputs[i]
            diff_min_arg = np.argsort(np.sum((centers - cur_client) ** 2, axis=1))
            for k_id in diff_min_arg:
                # 未满，入簇
                if cluster_inter_cur[k_id] < cluster_inter_max[k_id]:
                    cluster_inter_cur[k_id] += 1
                    new_labels[i] = k_id
                    break
                else:
                    continue  # 满了，找下一个
        # M：更新簇中心
        new_centers = np.zeros_like(centers, dtype=int)
        for i in range(k):
            cluster = inputs[new_labels == i]
            if cluster.shape[0] == 0:
                new_centers[i] = centers[i]
            else:
                new_centers[i] = np.mean(cluster, axis=0)
        # 退出条件：labels 不变，簇中心不比安
        if np.array_equal(labels, new_labels) and np.array_equal(new_centers, centers):
            break
        labels = new_labels
        centers = new_centers
    return labels, centers


labels, centers = kmeans(clients, k)
centers = centers.tolist()
centers.sort()
# 输出簇中心
for points in centers:
    print(*points.flatten())
# 预测新客户
client_pre = np.argmin(np.sum((centers - new_client) ** 2, axis=1))
print(client_pre + 1)

```

## 1217
### 第二题：最小二乘法/批量梯度下降
1. **梯度下降**：归一化！归一化！y_pred在回归问题中也需要通过`mean_y`和`std_y`还原
2. **最小二乘法** 闭式解




```python
k = int(input())
raw_train = list(map(float, sys.stdin.readline().split()))
train_data = np.array(raw_train).reshape((k, 4))
train_x, train_y = train_data[:, :3], train_data[:, -1].reshape((k, 1))

n = int(input())
raw_test = list(map(float, sys.stdin.readline().split()))
test_x = np.array(raw_test).reshape((n, 3))
# 归一化
mean_x, std_x = train_x.mean(axis=0), train_x.std(axis=0)
mean_y, std_y = train_y.mean(axis=0), train_y.std(axis=0)
train_x = (train_x - mean_x) / std_x
train_y = (train_y - mean_y) / std_y
test_x = (test_x - mean_x) / std_x

# W = np.zeros((3, 1))
# u_x, u_y = np.mean(train_x, axis=0), np.mean(train_y, axis=0)
# train_x -= u_x
# train_y -= u_y
# for i in range(3):
#     a1 = np.sum(train_x[:, i] * train_y)
#     a2 = np.sqrt(np.sum(train_x[:, i] ** 2) * np.sum(train_y**2))
#     W[i] = a1 / a2
# b = u_y - u_x @ W
# y_pred = test_x @ W + b
# print(*(int(v) for v in y_pred.flatten()))

W = np.zeros((3, 1))
b = np.zeros((1, 1))
lr = 0.01
y_pre_old = np.zeros((n))

epoch = 0


def predict_stable(y1, y2):
    y3 = y1 * 1000 // 1
    y4 = y2 * 1000 // 1
    return np.array_equal(y3, y4)


while True:
    # 前向传播
    z = train_x @ W + b
    dz = z - train_y
    # 梯度更新
    dw = train_x.T @ dz / k
    db = np.sum(dz) / k
    W -= lr * dw
    b -= lr * db
    # 预测是否稳定
    y_pre = test_x @ W + b
    y_pre_new = y_pre * std_y + mean_y
    if predict_stable(y_pre_old, y_pre_new):
        break
    y_pre_old = y_pre_new

print(*(int(v) for v in y_pre_new.flatten()))

```

# 常见模板
## 并查集

## 层序遍历的序列化和反序列化
1. deque直接存`node`，不需要维护nodes，建树完成直接返回root就行
2. null节点统一成空,也就是Node()


```python
def build_tree(data_list):
    if not data_list:
        return None
    root = Node(int(data_list[0]))
    q = deque([root])
    i = 1
    n = len(data_list)
    while q and i < n:
        parent = q.popleft()
        if i < n:
            if data_list[i] != "null":
                parent.left = Node(int(data_list[i]))
                q.append(parent.left)
            i += 1
        if i < n:
            if data_list[i] != "null":
                parent.right = Node(int(data_list[i]))
                q.append(parent.right)
            i += 1
    return root


def serilize(root):
    if not root:
        return "[]"
    res = []
    q = deque([root])
    while q:
        node = q.popleft()
        if node:
            res.append(str(node.val))
            q.append(node.left)
            q.append(node.right)
        else:
            res.append("null")
    while res and res[-1] == "null":
        res.pop()
    return "[" + ",".join(res) + "]"
```


```python
"""
	主要看 find 和 union 
    算法：合并<name , phone>   相同的phone,合并name(取 ASCII 更大的姓名)
"""

def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]


def union(parent, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        # 比较字典序，小的留下
        if rootX < rootY:
            parent[rootY] = rootX
        else:
            parent[rootX] = rootY


if __name__ == "__main__":
    n = int(input())
    # union-find 维护 name :根据重电话号码进行union
    parent = {}
    # data存储原始数据 [name : [number...]]
    data = {}
    phone_owner = {}

    for _ in range(n):
        line = list(sys.stdin.readline().split())
        name, phones = line[0], line[1:]
        if name not in parent:
            parent[name] = name
        for p in phones:
            #查找是否需要合并name
            if p in phone_owner:
                union(parent,name,phone_owner[p])
            else:
                phone_owner[p] = name
        data[name].add(phones)

    # 根据 union-find结果合并 name - [number1,...] 去重number
    res = defaultdict(set)
    for k,v in data.items():
        root = find(parent,k)
        for p in v:
            res[root].add(int(p))

    # number 排序
    for v in res.values():
        v.sort()

    # name 排序
    res_name = [x for x in res.keys()]
    res_name.sort()

    for x in res_name:
        print(x," ".join(map(str,res[x])))

```

## 卷积


```python
import numpy as np

def conv2d_basic(inputs, kernel, stride=1, padding=0, dilation=1):
    """
    2D卷积基础实现
    
    参数:
    inputs: 输入特征图 [C_in, H, W] 或 [B, C_in, H, W]
    kernel: 卷积核 [C_out, C_in, K_h, K_w]
    stride: 步长
    padding: 填充大小
    dilation: 膨胀系数
    """
    # 处理batch维度
    No_batch = False
    if np.ndim(inputs) == 3:
        inputs = inputs[np.newaxis, ...]
        No_batch = True

    B, C_in, H_in, W_in = inputs.shape
    C_out, C_in_k, K_h, K_w = kernel.shape

    assert C_in == C_in_k
    H_out = (H_in + 2 * padding - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W_in + 2 * padding - dilation * (K_w - 1) - 1) // stride + 1

    # 填充padding
    if padding > 0:
        inputs = np.pad(
            inputs,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )
    Output = np.zeros((B, C_out, H_out, W_out))
    for b in range(B):
        for c in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride

                    if dilation > 1:
                        h_indices = h_start + np.arange(K_h) * dilation
                        w_indices = w_start + np.arange(K_w) * dilation
                        roi = inputs[b, :, h_indices[:, None], w_indices]
                    else:
                        h_end = h_start + K_h
                        w_end = w_start + K_w
                        roi = inputs[b, :, h_start:h_end, w_start:w_end]
                    cur_out = np.sum(roi * kernel[c])
                    Output[b, c, h, w] = cur_out
    if No_batch:
        Output = Output[0]
    return Output

```

## 预测类问题(2分类、多分类、回归)
1. **核心思路：**前向传播(线性层+激活函数) -> (计算损失) -> 反向传播(梯度计算dw db) -> 参数更新
2. 组件：
	1. 线性层:`z = x@w + b`
		- w:二分类、回归`(n_features, 1)`；多分类`(n_features, n_classes)`
		- b:二分类、回归`(1,)`；多分类`(1, n_classes)`
	2. 激活函数:`a = f(z)`
		- Sigmoid(二分类): `1 / (1 + np.exp(-z))`
		- Softmax(多分类): `exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)); softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)`
		- ReLU: `np.maximum(0, z)`
	3. 损失函数:
		- 交叉熵损失(分类): `-np.sum(y_true * np.log(y_pred + 1e-15)) / m`
		- 均方误差(回归): `np.sum((y_true - y_pred) ** 2) / m`
	4. 反向传播(计算梯度)
		- 二分类交叉熵+Sigmoid:
			- `dz = a - y_true`
			- `dw = x.T @ dz / m`
			- `db = np.sum(dz) / m`
		- 多分类交叉熵+Softmax:
			- `dz = a - y_true`
			- `dw = x.T @ dz / m`
			- `db = np.sum(dz, axis=0, keepdims=True) / m`
		- 回归+ReLU:
			- `dz = (a - y_true) * (z > 0)`
			- `dw = x.T @ dz / m`
			- `db = np.sum(dz) / m`
	5. 参数更新: lr一般取0.01-0.1
		- 随机梯度下降sgd: `w -= lr * dw; b -= lr * db`
3. 框架


```python
import numpy as np
lr = 0.01
epochs = 1000
batch_size = 16
for epoch in range(epochs):
	for i in range(0,n_samples,batch_size):
		X_batch = X[i:i + batch_size]
		Y_batch = Y[i:i + batch_size]

		cur_batch_len = X_batch.shape[0]
		# 前向传播
		z = X_batch @ w + b
		a = activation_function(z)

		# 计算损失（可选）
		loss = compute_loss(Y_batch, a)

		# 反向传播
		dz = a - Y_batch  # 只有回归需要乘(z > 0)
		dw = X_batch.T @ dz / cur_batch_len
		db = np.sum(dz) / cur_batch_len # 只有多分类需要axis=0和keepdims

		# 参数更新
		w -= lr * dw
		b -= lr * db

	if (epoch + 1) % 150 == 0:
		lr *= 0.9  # 轻微衰减
```

## 预测类问题：最小二乘法
1. 闭式解$W^* = (X^TX)^{-1}X^TY$
2. 可以将b并到X、W中：X加一列1`np.c_`，W加一行`np.r_`。
3. 矩阵求逆：`np.linalg.pinv()`


```python
import numpy as np

X_b = np.c_[np.ones((len(X),1)),X]
W_b = np.linalg.pinv(X_b.T@X_b) @ X_b.T @ y

```

## 小总结
1. 数据读取
    - 一排读入时，使用`reshape()`：*`reshape((-1,1))`里的-1会被自动替换成相应的值*
    - 所有的向量(标签，预测值z)必须换成`n*1`的矩阵：`reshape((-1,1))`
2. 归一化  *题目SGD就得做，不然梯度会爆炸NAN*
    - `train_x`:`mean_x`、`std_x`
    - `train_y`:回归问题要做`(y -mean_y)/std_y`，分类问题不做
    - `test_x`:用训练集`mean_x`、`std_x`归一化，**对于回归问题：y_pre = y_pre \* std_y + mean_y**
3. 特征工程（多项式回归） 
    - 示例：$y = ax^2 + bx + c$ -> 构造新特征$x_1 = x, x_2 = x^2$
4. 正则化 *"防止过拟合”或者“L2惩罚系数λ"*
    - ![image.png](attachment:image.png)
5. 梯度求解：
    - ![image-2.png](attachment:image-2.png)

6. 梯度问题
    - 数据归一化？
    - lr太大了（试试0.01 0.001）？
    - 梯度有没有/N

7. **关键词**
    - 解析解、最小二乘、矩阵求解：`X`加一列，套闭式解
    - 距离、相似度、推荐、最近邻：
        - 欧氏距离：`linalg.norm()`
        - 余弦相似度：`a*b/(norm(a)*norm(b))`
    - 统计特征/概率：`np.mean()`, `np.std()`, `np.var()`
    - 正则化（一般配合梯度下降）
        - L2的导数`W`,L1的导数`sign(W)`


```python
import numpy as np
import sys

# === 0. 准备工具函数 ===
def sigmoid(z):
    # 数值稳定写法，防止 exp 溢出
    return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)), 
                    np.exp(z) / (1 + np.exp(z)))

def softmax(z):
    # 多分类用，减去 max 防止溢出
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# === 1. 核心训练类 (适用一切) ===
class LinearModel:
    def __init__(self, mode='regression', lr=0.1, epochs=1000, lam=0):
        self.mode = mode  # 'regression' 或 'binary' 或 'multi'
        self.lr = lr
        self.epochs = epochs
        self.lam = lam  # 正则化系数
        self.W = None
        self.b = None
        self.y_mean = 0
        self.y_std = 1
        self.x_mean = 0
        self.x_std = 1
        
    def fit(self, X, y):
        # --- A. 归一化 (X 必须做) ---
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0) + 1e-8 # 加上极小值防止除0
        X_norm = (X - self.x_mean) / self.x_std
        
        # --- B. 归一化 (y 只有回归由于量级差异大建议做) ---
        y = y.reshape(-1, 1) # 强转 (N, 1)
        if self.mode == 'regression':
            self.y_mean = np.mean(y)
            self.y_std = np.std(y) + 1e-8
            y_norm = (y - self.y_mean) / self.y_std
        else:
            y_norm = y # 分类问题 label 0/1 不需要归一化

        # --- C. 初始化 ---
        N, D = X_norm.shape
        # 如果是多分类，W是 (D, K)，y是One-hot (N, K)
        # 这里默认写二分类/回归 (D, 1)
        self.W = np.zeros((D, 1)) 
        self.b = np.zeros((1, 1))
        
        # --- D. 梯度下降循环 ---
        for i in range(self.epochs):
            # 1. Forward
            z = X_norm @ self.W + self.b
            
            if self.mode == 'regression':
                pred = z
            elif self.mode == 'binary':
                pred = sigmoid(z)
                
            # 2. Error (核心：万能公式预测-真实)
            error = pred - y_norm
            
            # 3. Gradients
            # dW = (1/N) * X.T @ error + (lam/N) * W
            dw = (X_norm.T @ error) / N + (self.lam / N) * self.W
            db = np.sum(error) / N
            
            # 4. Update
            self.W -= self.lr * dw
            self.b -= self.lr * db
            
            # (可选) 打印 loss 监控收敛
            if i % 1000 == 0:
                pass 

    def predict(self, X_test):
        # 别忘了测试集也要归一化！使用训练集的参数
        X_test_norm = (X_test - self.x_mean) / self.x_std
        z = X_test_norm @ self.W + self.b
        
        if self.mode == 'regression':
            # 还原数值
            return z * self.y_std + self.y_mean
        elif self.mode == 'binary':
            # 输出概率
            return sigmoid(z)
```

## 朴素贝叶斯 (Naive Bayes) 
核心思路：`P(Class|Doc) ∝ P(Class) * Π P(Word|Class)`
常考点：
- 词频统计：使用 `Counter` 或 `defaultdict`。
- 对数空间计算：防止多个小概率相乘导致下溢 (Underflow)，乘法变加法。
- 平滑处理 (Laplace Smoothing)：
	- **文本分类P(word|class):** 分子+1，分母+V (V为词表大小)。
	- **属性离散P($x_i$|class)** 分子+1，分母+$V_c$(该属性取值个数)  用`len(np.unique(X[:,i]))`计算
	- **连续属性P($x_i$|class)** 计算c的均值`mean`和方差`var`，$\log P(x|c) = -\frac{1}{2} \log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$  方差需要加上`1e-9`防止除0


```python
# 1.属性离散
import numpy as np
from collections import defaultdict,Counter

class NaiveBayes:
	def __init__(self):
		self.class_log_prior = {}	# P(class)
		self.feature_log_prob = defaultdict(lambda :defaultdict(dict)) # P(fi|c)
	def fit(self,x,y):
		# 基础统计
		n_sample,n_features = x.shape
		y_cnt = Counter(y) 
		
		# 计算先验概率 P(class)
		for k,v in  y_cnt.items():
			self.class_log_prior[k] = np.log(v/n_sample)
		
		# 统计特征计数 count(class = c, f_i = val)
		category_x_num =  defaultdict(lambda :defaultdict(Counter()))
		for sample_features,sample_label in zip(x,y):
			for f_idx,f_val in enumerate(sample_features):
				category_x_num[sample_label][f_idx][f_val] += 1
		
		# 计算条件概率P(xi|c)
		for c in self.class_log_prior.keys(): 
			for f_id in range(n_features):
				# 类别c里，属性i的所有(fi_取值,个数)
				tf_counter = category_x_num[c][f_id]

				n_vocab = len(np.unique(x[:,f_id]))
				n_c = sum(tf_counter.values())
				default_prob = np.log(1/(n_vocab+n_c))
				self.feature_log_prob[c][f_id][None] = default_prob
				for x_val in np.unique(x[:,f_id]):
					val = np.log((tf_counter[x_val] + 1)/(n_c+n_vocab))
					self.feature_log_prob[c][f_id][x_val] = val
	def predict(self,data):
		predicts = []
		for sample in data:
			best_label,best_prob = None,float('-inf')
			for cur_label,pc in self.class_log_prior.items():
				cur_prob = pc
				for idf,f_val in enumerate(sample):
					feat_dict = self.feature_log_prob[cur_label][idf]
					cur_prob += feat_dict.get(f_val,feat_dict[None])
				if cur_prob >= best_prob:
					best_prob = cur_prob
					best_label = cur_label
			predicts.append(best_label)

		return predicts



```


```python
# 2. 属性连续  注意var需要加上1e-9 防止除0
import numpy as np
import math
from collections import defaultdict,Counter

class NaiveBayes:
	def __init__(self):
		self.class_prior = {} # P(c)
		self.feature_prob = defaultdict(lambda:defaultdict(list))#P(x|c)
	
	def fit(self,x,y):
		# 基础信息
		n_samples, n_features = x.shape
		
		# 1. P(c)
		y_label = Counter(y)
		for label_id,label_n in y_label.items():
			self.class_prior[label_id] = np.log(label_n/n_samples)

		# 2. P(x|c) ：计算u_c var_c
		for label_id in y_label.keys():
			for f_id in range(n_features):
				cur_data = x[y==label_id,f_id]
				u_c, var_c = np.mean(cur_data),np.var(cur_data) + 1e-9
				self.feature_prob[label_id][f_id] = [u_c,var_c]
	def predict(self,x):
		predicts = []
		for sample in x:
			best_label, best_prob = None, float('-inf') 
			for label_id,pc in self.class_prior.items():
				cur_prob = pc
				for f_id,f_val in enumerate(sample):
					u_var = self.feature_prob[label_id][f_id]
					u_c,var_c = u_var[0],u_var[1]
					cur_prob += -0.5*math.log(2*math.pi*var_c)-(f_val - u_c)**2/(2*var_c)
				if cur_prob > best_prob:
					best_label = label_id
					best_prob = cur_prob
			predicts.append(best_label)
		return predicts
```


```python
# 3. 文本类 注意下初始化defaultdict(defaultdict(Counter))  没有()!
import numpy as np
from collections import defaultdict,Counter
class NaiveBayes:
	def __init__(self):
		self.class_prob = {}
		self.words_category_prob = defaultdict(lambda : defaultdict(float))
		self.words_dict = set()
	def fit(self,x,y):
		# 基础信息
		n_samples = x.shape[0]
		
		# 1. P(c)
		y_label = Counter(y)
		for label_id,label_num in y_label.items():
			self.class_prob[label_id] = np.log(label_num/n_samples)
		
		# 2. P(word|c)
		word_c_num = defaultdict(Counter) # [c:(w1:n1,w2,n2)]
		for cur_doc,cur_label in zip(x,y):
			for word in cur_doc:
				word_c_num[cur_label][word] += 1
				self.words_dict.add(word)
		V = len(self.words_dict)
		for label_id in y_label.keys():
			n_c = np.sum(word_c_num[label_id].values())
			default_prob = np.log(1/(n_c + V))
			self.words_category_prob[label_id][None] = default_prob
			for word,word_n in word_c_num[label_id].items():
				val = np.log((word_n+1)/(n_c + V))
				self.words_category_prob[label_id][word] = val
	def predict(self,x):
		predicts = []
		for sample in x:
			best_label,best_prob = None,float('-inf')
			for label_id,pc in self.class_prob.items():
				cur_prob = pc
				defeat_data = self.words_category_prob[label_id]
				for word in sample:
					cur_prob += defeat_data.get(word,defeat_data[None])
				if cur_prob > best_prob:
					best_label = label_id
					best_prob = cur_prob
			predicts.append(best_label)
		return predicts
		
```

## K-Means 聚类
核心思路：
1. 初始化中心 (Random Choice)。
2. **E步**：计算每个点到K个中心的距离，分配到最近簇。
3. **M步**：重新计算每个簇的平均值作为新中心。
4. 迭代直到收敛 (中心变化 < tol)。

技巧：
- 利用广播计算距离矩阵 `(N, 1, D) - (1, K, D)` 得到 `(N, K, D)`，再 norm 得到 `(N, K)`。
- 注意处理空簇 (Empty Cluster) 的情况。


```python
import numpy as np

def Kmeans(x, k, max_iter=1000, tol=1e-4):
    n_samples, n_features = x.shape
    # 1. 随机选取中心
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = x[indices, :]

    for _ in range(max_iter):
        # E: 计算dist,确定新label
        distance = np.zeros((n_samples, k))  # m * k
        for i in range(k):
            cur_point = centroids[i]
            diff = x - cur_point
            dist = np.sqrt(np.sum(diff**2, axis=1))
            distance[:, i] = dist
        label = np.argmin(distance, axis=1)

        # M：更新中心
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_point = x[label == i]
            cur_point = np.mean(cluster_point, axis=0)
            if cluster_point.shape[0] != 0:  # 存在新的样本
                new_centroids[i] = cur_point
            else:
                new_centroids[i] = centroids[i]
        # 计算是否需要终止
        diff_c = new_centroids - centroids
        centroids = new_centroids
        if np.sum(np.sqrt(np.sum(diff_c**2, axis=1))) < tol:
            break
        
    return label, centroids
	
```

## ID3 / CART 决策树构建
核心思路：递归地寻找最佳分裂属性，直到满足终止条件（纯度足够高或达到深度）。

分裂指标：
1. **信息增益 (ID3)**: 使用 **信息熵 (Entropy)**。
   - $Entropy(D) = - \sum p_i \log_2 p_i$
   - $Gain(D, A) = Entropy(D) - \sum \frac{|D_v|}{|D|} Entropy(D_v)$
2. **基尼系数 (CART)**: 使用 **Gini Impurity**。
   - $Gini(D) = 1 - \sum p_i^2$
   - 越小越纯。计算简单（无对数），常用于分类树。

常考点：手算 Entropy/Gini，递归代码结构。


```python
# 如果有depth 要求，就在build_tree里面加一个depth参数，在开始就判断是否需要终止 -> 直接变成叶节点
import numpy as np
from collections import Counter

class Node:
	def __init__(self,fi=None,th=None,label=None,l=None,r=None):
		self.fi = fi
		self.th = th
		self.label = label
		self.l = l
		self.r = r
def cal_gini(y):
	if len(y) == 0:
		return 0
	y_label = Counter(y)
	n_sample = len(y)
	ans = 1 - sum( (v/n_sample)**2 for v in y_label.values())
	return ans
def cal_entropy(y):
	if len(y) == 0:
		return 0
	y_label = Counter(y)
	n_sample = len(y)
	ans = 0
	for v in y_label.values():
		p = v/n_sample
		ans += -p*math.log2(p)
	return ans

def most_common_labels(y):
	y_cnt = Counter(y)
	max_c = max(y_cnt.values())
	candidates = [k for k, v in y_cnt.items() if v == max_c]
	return min(candidates)

def build_tree(X,y):
	# 1. 终止条件：不纯度为0/没有feature可分
	if len(np.unique(y)) == 0:
		return Node(label = y[0])  

	m,n = X.shape
	best_gini = float('inf')  #target:找到最小值(越小越纯)
	best_split = None

	# 2. 遍历feature找最佳划分
	for f in range(n):
		feature_vals = np.unique(X[:,f])
		for th in feature_vals:
			left_idx = X[:,f] <= th
			y_l = y[left_idx]
			y_r = y[~left_idx]
			if len(y_l) == 0 or len(y_r) == 0:
				continue  #切分无效
			gini = (len(y_l)* cal_gini(y_l) + len(y_r)*cal_gini(y_r))/m
			if gini < best_gini:
				best_gini = gini
				best_split = (f,th,y_l,y_r)
	# 3. 无法划分，返回叶节点
	if best_split is None:
		common_val = most_common_labels(y)
		return Node(label=common_val)
	
	# 4. 递归划分
	f_idx ,th_val,y_l,y_r = best_split
	left_node = build_tree(X[X[:,f_idx]<=th_val],y_l)
	right_node = build_tree(X[X[:,f_idx]>th_val],y_r)
	return Node(fi=f_idx,th=th_val,l=left_node,r=right_node)

```

## IoU / NMS (目标检测)
核心思路：
- **IoU**: 交集面积 / 并集面积。
- **NMS**: 按置信度排序，选最高的，删除与其IoU > 阈值的，循环。

注意坐标格式：`(x1, y1, x2, y2)`。
面积计算：`(x2 - x1) * (y2 - y1)` (如果不包含边界+1的话)。机考通常假设点坐标，可能需要+1，视题目而定（通常 `x2-x1` 即可）。


```python
import numpy as np

def calculate_iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """
    # 1. 计算交集区域
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])
    
    # 2. 如果不相交，宽/高为0
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter_area = w * h
    
    # 3. 计算并集区域
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Union = A + B - Inter
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: list of [x1, y1, x2, y2]
    scores: list of scores
    """
    # 转换为numpy方便操作
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    keep = []
    # 1. 按分数降序排列
    idxs = scores.argsort()[::-1]
    
    while len(idxs) > 0:
        # 2. 选出分数最高的框
        current = idxs[0]
        keep.append(current)
        
        if len(idxs) == 1: break
            
        # 3. 计算当前框与其他剩余框的IoU
        # 这里用向量化计算，或者由于数量不多，循环算便于记忆
        remaining_idxs = idxs[1:]
        ious = []
        for rest in remaining_idxs:
            ious.append(calculate_iou(boxes[current], boxes[rest]))
        
        ious = np.array(ious)
        
        # 4. 保留IoU小于阈值的框（即去除重叠度高的）
        # ious < thresh 得到的是 remaining_idxs 中的索引掩码
        # remaining_idxs[ious < thresh] 得到的是原始索引
        idxs = remaining_idxs[ious < iou_threshold]
        
    return keep, boxes[keep]

# Usage:
# boxes = [[10, 10, 50, 50], [12, 12, 48, 48]]
# scores = [0.9, 0.8]
# keep_indices, result = nms(boxes, scores)
```

## DP问题
路径 DP（比如：机器人走格子，最小路径和）。
背包问题（01背包，完全背包）。
打家劫舍系列（当前选还是不选）。
最长公共子序列/子串（LCS）

常规的“最大/最小面积”题：
通常是用 单调栈（如：直方图最大矩形）。
或者是 DP（如：最大正方形）。
或者是 双指针（如：盛最多水的容器）。

## LSTM 实现
1. lstm step 和 循环
2. format_output:`s = f"{ans:.3f}"`,`s.rstrip("0")`,`if '.' in s:`.
3. tanh 可以直接用公式


```python
# x_t :(7,) h_prev :(5,) W_ :(5,12)
def lstm_step(x_t, h_prev, c_prev, W_f, b_f, W_i, b_i, W_g, b_g, W_o, b_o):
    concat_input = np.concatenate((h_prev, x_t))
    f_t = sigmoid(W_f @ concat_input + b_f)
    i_t = sigmoid(W_i @ concat_input + b_i)
    g_t = tanh(W_g @ concat_input + b_g)
    o_t = sigmoid(W_o @ concat_input + b_o)
    c_t = c_prev * f_t + i_t * g_t
    h_t = tanh(c_t) * o_t
    return h_t, c_t

output_res = []
for t in range(sequence_length):
    x_t = inputs[t]
    h_t, c_t = lstm_step(x_t, h_t, c_t, W_f, b_f, W_i, b_i, W_g, b_g, W_o, b_o)
    output_res.append(h_t[0])


```

# 机试tips
## ACM OJ报错
```python 
# Check Point 1: 树的大小不对？ -> 让他 RE
if len(nodes) != N:
    print(1 / 0)  # 信号：Runtime Error

# Check Point 2: 中间逻辑非法？ -> 让他 TLE (超时)
if some_val < 0:
    while True: pass  # 死循环，信号：Time Limit Exceeded

# Check Point 3: 最终结果不对？ -> 让他 WA (答案错误)
if ans == -1:
    print("Wrong Answer" + "a" * 1000) # 输出一个明显的错乱结果，信号：Wrong Answer
    sys.exit(0) # 正常退出
```

## numpy 函数
1. **筛选出第i列所有取值可能：**`unique_val = np.unique(X_train[:,i])`

2. **按照某列的取值筛选:**`data[data[:,i] > k]`

3. **转成三角矩阵:** `np.triu()`转成上三角矩阵,`np.tril()`下三角矩阵

4. **处理图像分块:**`reshape()`和`transpose()`(按行读取!,请注意grid_h在p_size之前.)
```python
     # 拆分:(B,C,H,W) -> (B,C,grid_h,patch_size,grid_w,patch_size)
    step1 = images.reshape(B, C, grid_h, patch_size, grid_w, patch_size)
    # 调整维度 (b,c,gh,ps,gw,ps) -> (b,gh,gw,c,ps,ps)
    step2 = step1.transpose(0, 2, 4, 1, 3, 5)
    # flatten 也能用reshape实现!
    patch_flat_dim = patch_size * patch_size * C
    step3 = step2.reshape(B, num_patches, patch_flat_dim)

```
5. 拼接数组:`np.concatenate((a,b), axis=0)`  注意axis。行拼接`np.r_`，列拼接`np.c_`

6. 数学运算：
    - 求伪逆矩阵：`np.linalg.pinv(A)`
    - 求范数:`np.linalg.norm(A)` ,求欧氏距离`np.linalg.norm(a - b)`
    - 截断数值，防止`log(0)`或者数值爆炸  `np.clip(A, min,max)`
        - `np.log(np.clip(A,1e-15,1.0))`
    - 四舍五入: `np.round(A)`

7. 高级索引mask
    - `A[A>0]`
    - `np.where(condition, x, y)`
        - 写`sigmoid`防止溢出: `np.where(z > 0, 1/(1+ np.exp(-z)), np.exp(z)/(np.exp(z) + 1))`


## python 函数
1. **多层哈希：**`defaultdict(lambda: defaultdict(list))`
2. **lambda：**
	- **排序:** `A.sort(key = lambda x:x[1])`
	- **默认字典:** `defaultdict(lambda: defaultdict(int))`
3. **排序**:`idxs = scores.argsort()[::-1]`将scores按照值从大到小排序，返回**索引**

4. **字符与ascii码之间的转化：**`ascii_val = ord(ch)`，`ch = chr(ascii_val)`

# 机试 Tips

这里汇总了在 AI 机考和算法题中常用的 Python 原生函数和 Numpy 函数，以及易错点。

## Python 原生函数
常用的数据结构操作和内置函数。

## Numpy 函数
处理矩阵和数组的常用操作及坑点。


```python
import collections
import heapq

# --- 1. 排序与查找 ---
data = [("A", 3), ("B", 1), ("C", 2)]

# sort() vs sorted()
# sort() 是 list 的方法，原地修改，返回 None
data_copy = data[:]
data_copy.sort(key=lambda x: x[1], reverse=True) # 按第二个元素降序
print("List.sort:", data_copy)

# sorted() 是内置函数，返回新列表，不改变原列表
new_data = sorted(data, key=lambda x: x[1]) 
print("sorted():", new_data)

# --- 2. 计数与频率 ---
nums = [1, 2, 2, 3, 3, 3]
cnt = collections.Counter(nums)
# most_common(k) 返回出现次数最多的前 k 个元素，返回 list of (elem, count)
print("Most common:", cnt.most_common(2)) 
# 字典序最小的众数技巧: sorted(cnt.items(), key=lambda x: (-x[1], x[0]))

# --- 3. 堆 (Priority Queue) ---
# Python 默认是小顶堆
heap = [3, 1, 4, 1, 5]
heapq.heapify(heap) # O(N) 建堆
print("Heap pop:", heapq.heappop(heap)) # 弹出最小值 1
heapq.heappush(heap, 2)

# --- 4. 字符串处理 ---
s = "  Hello   World  "
print("Split:", s.strip().split()) # 去除首尾空格并按空白字符分割
# join
print("Join:", "-".join(["a", "b", "c"]))

# --- 5. 其他实用函数 ---
# zip: 并行遍历
names = ["Alice", "Bob"]
scores = [85, 90]
for n, s in zip(names, scores):
    print(n, s)

# any() / all(): 判断 iterable 中是否 有一个为真 / 全部为真
print(any([False, True, False])) # True
print(all([True, True, True]))   # True
```


```python
import numpy as np

# --- 1. 创建与形状 ---
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Shape:", arr.shape)
print("Reshape:", arr.reshape(3, 2))  # 不改变数据，只改变视图
# 增加维度 (常用于广播)
# (2, 3) -> (2, 1, 3)
expanded = np.expand_dims(arr, axis=1) 
# 或者使用 None 索引
expanded_v2 = arr[:, None, :] 

# --- 2. 索引与最值 ---
# argmin / argmax: 返回扁平化后的下标，除非指定 axis
flat_argmax = np.argmax(arr) 
col_argmax = np.argmax(arr, axis=0) # 每列最大值的索引
print("Argmax (flat):", flat_argmax)
print("Argmax (axis=0):", col_argmax)

# where: 寻找满足条件的坐标 或 三元运算符
indices = np.where(arr > 3) # 返回 tuple of arrays (row_indices, col_indices)
print("Where > 3:", list(zip(indices[0], indices[1])))
# np.where(condition, x, y) -> if cond then x else y
print("Where cond:", np.where(arr > 3, 1, 0))

# --- 3. 排序 ---
# np.sort 返回副本，arr.sort() 原地修改
# argsort: 返回排序后的索引 (非常常用！)
idx = np.argsort(arr, axis=1) # 对每一行进行排序的索引
print("Argsort axis=1:\n", idx)

# lexsort: 多键排序 (最后传入的主键)
# 比如先按第二列排，再按第一列排
# primary key is arr[:, 0] ?? NO, lexsort((secondary, primary))
keys = (arr[:, 1], arr[:, 0]) 
# 注意：lexsort 的键是反过来的，最后一个键是主键
# 例子：按第一列排序，如果第一列相同按第二列
idx_lex = np.lexsort((arr[:, 1], arr[:, 0])) 

# --- 4. 统计与计算 ---
# sum, mean, std
# Pitfall: keepdims=True
# Softmax 实现经常遇到不需要降维的情况
exp_x = np.exp(arr)
# keepdims=True 保持 (2, 1) 形状，方便后续广播除法
softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True) 

# array_equal: 检查两个数组是否形状和元素完全相同
print("Equal:", np.array_equal(arr, arr))
# allclose: 浮点数比较 (容忍误差)
print("Allclose:", np.allclose(1.0, 1.000000001))

# --- 5. 拼接与堆叠 ---
a = np.array([1, 2])
b = np.array([3, 4])
# concatenate: 沿现有维度连接
print("Concat:", np.concatenate([a, b])) # [1, 2, 3, 4]
# stack: 增加新维度连接
print("Stack:", np.stack([a, b])) # [[1, 2], [3, 4]]

# --- 6. 唯一值 ---
# unique: 返回排序后的唯一值
u, counts = np.unique([1, 1, 2, 2, 3], return_counts=True)
print("Unique counts:", dict(zip(u, counts)))
```
