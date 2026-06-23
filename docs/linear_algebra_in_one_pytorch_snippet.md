# 大学4年没讲明白的线性代数，被一段 PyTorch 代码讲透了

> 行列式、特征值、矩阵的秩、向量空间……当年背得滚瓜烂熟，考完试就全忘了。
> 因为老师只给了你**算法**，没给你**画面**。
>
> 这篇文章反过来：**不背一个定义，不证一个定理**，全部用能跑起来的 PyTorch 代码，
> 把线性代数最核心的那几个概念，一段一段"跑"给你看。
>
> 并且顺手回答四个让无数人卡住的灵魂拷问：
>
> - 人类识别的"特征"，和矩阵里的"特征值/特征向量"，到底是不是一回事？
> - 难道大模型、神经网络，说穿了就是在做**向量点积**？
> - 一个个**离散**的字、像素、整数，是怎么变成**连续可导**、能被梯度下降优化的东西的？
> - 都说训练靠 GPU，**GPU 是不是就是一台大型向量机**？

---

## 0. 一句话主线

如果只能留一句话，那就是这句：

> **线性代数不是"算数表"，是"搬空间"。一个矩阵就是一台把整个空间整体搬动的机器。**

整个深度学习里，你看到的所有 `@`、`matmul`、`nn.Linear`、`Q @ K.T`，
本质都是在反复做同一件事：**把数据从一个空间，搬到另一个更好处理的空间。**

```python
import torch

x = torch.tensor([1.0, 2.0])          # 一个点 / 一个向量
M = torch.tensor([[2.0, 0.0],
                  [0.0, 3.0]])         # 一台机器：横向拉 2 倍，纵向拉 3 倍

print(M @ x)                          # tensor([2., 6.]) —— 这个点被"搬"走了
```

记住这个画面，下面所有概念都挂在它上面。

---

## 1. 向量：不是"一列数字"，是"空间里的一个箭头"

课本第一页就给你一个列向量 `[1, 2, 3]^T`，然后让你背"加法、数乘"。
但向量真正的意思是：**从原点指出去的一根箭头**，它有方向、有长度。

```python
import torch

a = torch.tensor([3.0, 4.0])

length = torch.linalg.norm(a)         # 长度 = 勾股定理 = sqrt(3^2 + 4^2)
print("这根箭头的长度:", length.item())   # 5.0
```

两个向量之间最重要的关系是**点积**。点积衡量"两根箭头有多同向"：

```python
import torch

a = torch.tensor([1.0, 0.0])          # 指向正右
b = torch.tensor([0.0, 1.0])          # 指向正上
c = torch.tensor([1.0, 1.0])          # 指向右上 45°

print(torch.dot(a, b))   # 0.0  —— 垂直，毫不相关
print(torch.dot(a, c))   # 1.0  —— 有点同向
print(torch.dot(a, a))   # 1.0  —— 和自己最同向
```

**点积 = 0 就是"垂直/正交/无关"，点积大就是"方向相近"。**
记住这个直觉——它会在第 6 节"神经网络只是点积吗"里变成主角。

---

## 2. 矩阵就是映射：把空间整体搬动

别把矩阵看成"一堆数字排成方块"，把它看成"对空间里**所有点**施加的同一个动作"。
喂几个点进去，看它们被搬到哪：

```python
import torch

M = torch.tensor([[2.0, 0.0],
                  [0.0, 3.0]])         # 横向拉 2 倍，纵向拉 3 倍

points = torch.tensor([[1.0, 0.0],     # 右
                       [0.0, 1.0],     # 上
                       [1.0, 1.0]])    # 右上角
print(points @ M.t())
# [[2, 0], [0, 3], [2, 3]] —— 每个点都被同一条规则搬动
```

换一个矩阵，就是换一个动作。下面四台机器：旋转、剪切、投影、放大：

```python
import torch, math

theta = math.radians(30)
rotate  = torch.tensor([[math.cos(theta), -math.sin(theta)],   # 转 30°
                        [math.sin(theta),  math.cos(theta)]])
shear   = torch.tensor([[1.0, 1.0],     # 剪切：越往上越往右滑（像推一摞书）
                        [0.0, 1.0]])
project = torch.tensor([[1.0, 0.0],     # 投影：所有点拍扁到 x 轴
                        [0.0, 0.0]])

v = torch.tensor([1.0, 2.0])
print(rotate  @ v)   # 转了角度，长度不变
print(shear   @ v)   # x 被 y 拖着走
print(project @ v)   # y 变 0，二维塌成一维
```

把一整张网格喂进去，你就能**亲眼看见**空间被扭曲：

```python
import torch
import matplotlib.pyplot as plt

xs, ys = torch.meshgrid(torch.linspace(-2, 2, 11),
                        torch.linspace(-2, 2, 11), indexing="xy")
grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)

M = torch.tensor([[1.0, 0.8],
                  [0.3, 1.2]])
moved = grid @ M.t()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(grid[:, 0],  grid[:, 1]);  ax[0].set_title("原始空间")
ax[1].scatter(moved[:, 0], moved[:, 1]); ax[1].set_title("被矩阵 M 映射后")
plt.show()
# 横平竖直的网格，被整体"斜拉"成平行四边形网格
```

**这就是线性代数最该记住的一张图：直线还是直线、原点还是原点，但整个空间被均匀地拉斜了。**

---

## 3. 行列式 = 面积的缩放倍数

老师让你背"按行展开""余子式""代数余子式"，背完你还是不知道行列式**是什么**。
其实它简单到一句话：**行列式 = 这台机器把面积放大/缩小了几倍。**

取单位正方形（面积 1）的两条边，用矩阵搬动它们，搬完后围出的面积，正好等于行列式：

```python
import torch

M = torch.tensor([[2.0, 1.0],
                  [0.0, 3.0]])

e1 = torch.tensor([1.0, 0.0])
e2 = torch.tensor([0.0, 1.0])
a, b = M @ e1, M @ e2                 # 两条边被搬动后

area = abs(a[0] * b[1] - a[1] * b[0]) # 平行四边形面积
print("搬动后面积:", area.item())          # 6.0
print("行列式:    ", torch.det(M).item())  # 6.0 —— 完全一样
```

- `det = 2`：面积变 2 倍。
- `det = 1`：面积不变（旋转就是，它只转不拉）。
- `det = 0`：**面积被压成 0**，整个平面被拍扁成一条线，信息丢了，再也回不来。

`det = 0` 就是课本里那个抽象的"矩阵不可逆/奇异"。它一点不抽象——
**就是你把立体拍成了影子，没法从影子反推回立体。**

```python
import torch
project = torch.tensor([[1.0, 0.0],
                        [0.0, 0.0]])   # 拍扁到 x 轴
print(torch.det(project).item())       # 0.0 —— 拍扁了，不可逆
```

---

## 4. 特征向量：映射里"不改方向"的那几根轴

这是线性代数最被神化、也最被讲烂的概念。其实就一句话：

> 大多数点被矩阵搬动后**方向会变**；但总有那么几根特殊方向，搬动后**方向不变，只是被拉长或缩短**。
> 这几根方向就是**特征向量**，拉伸的倍数就是**特征值**。

代码找出来并验证：

```python
import torch

M = torch.tensor([[2.0, 1.0],
                  [1.0, 2.0]])
vals, vecs = torch.linalg.eig(M)
print("特征值:", vals.real)            # [1., 3.]

v = vecs[:, 1].real                    # 取特征值 3 对应的特征向量
print("M @ v :", M @ v)                # 等于 3*v
print("3 * v :", 3 * v)                # 方向完全一样，只是长度 ×3
```

`M @ v` 和 `3 * v` 一模一样：方向没动，只是被拉长了 3 倍。
**特征向量是这台机器的"骨架方向"。** 任何复杂的线性变换，
都可以理解成"沿着这几根骨架，各自拉伸不同倍数"。PCA 降维、协方差分析、PageRank，全靠它。

---

## 🤔 疑惑点一：人类识别的"特征"，和矩阵的"特征值/特征向量"是一回事吗？

这是中文翻译挖的一个大坑。两个"特征"，英文根本是两个词：

| 中文 | 英文 | 含义 |
|---|---|---|
| 人脸有"高鼻梁"这个**特征** | **feature** | 数据的一个可观测属性、一个维度 |
| 矩阵的**特征**值/向量 | **eigen**value / **eigen**vector | 线性变换里方向不变的那根轴 |

`eigen` 是德语，意思是"自身的、固有的"。所以 eigenvector 更准确的翻译是
"**本征向量**"——这台变换机器**自身固有**的骨架方向，和"人脸特征"没有半点关系。

但它们之间有一座桥，叫 **PCA（主成分分析）**。
PCA 干的事，正是：**用矩阵的特征向量，去找出数据里最重要的那几个"人类意义上的特征方向"。**
代码看一眼这座桥：

```python
import torch

# 造一批"又高又重"强相关的人：身高和体重几乎成正比
torch.manual_seed(0)
h = torch.randn(500) * 10 + 170        # 身高
w = h * 0.6 + torch.randn(500) * 2     # 体重 ≈ 0.6*身高 + 噪声
data = torch.stack([h, w], dim=1)
data = data - data.mean(0)             # 去均值（PCA 第一步）

# 协方差矩阵的"特征向量" = 数据散布得最开的方向
cov = (data.T @ data) / len(data)
vals, vecs = torch.linalg.eigh(cov)
print("两个方向上的方差(特征值):", vals)        # 一大一小
print("最大特征值对应的方向:", vecs[:, -1])     # 指向"又高又重"那条对角线
```

输出里那个**最大特征值**对应的特征向量，会指向身高体重数据散得最开的那条 45° 对角线。
这条线，人类会给它起个名叫"**体型大小**"——

**这就是两个'特征'唯一的相遇点：矩阵的 eigenvector（数学骨架）帮你算出了
人类能命名的 feature（语义特征）。** 算法找轴，人类起名，分工明确。

> 顺便说：神经网络里 `nn.Linear` 提取的"特征"，是 feature 那个意思——
> 它在学一组**新的坐标方向**，把数据投影上去，量出"输入在每个方向上有多少分量"。
> 见下一节。

---

## 5. nn.Linear 就是一次矩阵映射（线性代数和深度学习的接头）

现在把前四节的画面，对接到 PyTorch 最基础的 `nn.Linear`：

```python
import torch
import torch.nn as nn

x = torch.tensor([[2.0, 3.0]])         # 1 个样本，2 个特征
layer = nn.Linear(2, 1)                # 2 维输入 -> 1 维输出
W, b = layer.weight, layer.bias        # W 形状 (1,2)，b 形状 (1,)

manual = x @ W.t() + b                 # PyTorch 内部做的事
print(torch.allclose(layer(x), manual))  # True
```

`nn.Linear(2, 1)` 就是**一次矩阵乘法加偏置**，一点不神秘。
看高维一点的——`nn.Linear(3, 128)`：

```python
import torch.nn as nn
layer = nn.Linear(3, 128)
print(layer.weight.shape)              # torch.Size([128, 3])
```

它是一个 `128 × 3` 的矩阵，把 **3 维输入映射到 128 维空间**。
矩阵的**每一行**是一个 3 维方向；输入和这一行做点积，就是**把输入投影到这个方向上**，
量出"输入在这个方向有多少分量"。**128 行 = 从 128 个角度同时打量同一个输入。**

为什么要升维？因为**在低维挤成一团、分不开的数据，升到高维往往一刀就能切开**：

```python
import torch

# 同心内外圈，在 2D 平面无法用一条直线分开
theta = torch.linspace(0, 2 * 3.14159, 100)
inner = torch.stack([torch.cos(theta), torch.sin(theta)], 1) * 1.0
outer = torch.stack([torch.cos(theta), torch.sin(theta)], 1) * 2.0

def lift(p):                           # 升维：加一维 z = x^2 + y^2（半径平方）
    z = (p ** 2).sum(1, keepdim=True)
    return torch.cat([p, z], 1)

print("内圈 z 均值约:", lift(inner)[:, 2].mean().item())   # ≈ 1
print("外圈 z 均值约:", lift(outer)[:, 2].mean().item())   # ≈ 4
# 升到 3D 后，内外圈在 z 轴上彻底分开，一个平面 z=2.5 就切开了
```

**`nn.Linear` 升维 + 激活函数折弯，合起来就干一件事：把搅在一起的数据，
搬到一个能被简单切开的新空间。** 训练，就是在学"往哪个空间搬"。

---

## 🤔 疑惑点二：大模型、神经网络，说穿了就是在做"向量点积"吗？

**大体上，是的——但"点积"这两个字撑起了整座大厦。** 我们一层层拆。

**第一层：单个神经元 = 一次点积。**

```python
import torch

x = torch.tensor([0.5, 0.2, 0.9])      # 输入向量
w = torch.tensor([1.0, -2.0, 0.5])     # 一个神经元的权重
b = 0.1
print("神经元输出:", torch.dot(x, w) + b)   # 一次点积 + 偏置，仅此而已
```

**第二层：一个 Linear 层 = 一堆点积打包成矩阵乘法。**

```python
import torch
x = torch.randn(4, 3)                  # 4 个样本，每个 3 维
W = torch.randn(8, 3)                  # 8 个神经元，每个 3 维权重
out = x @ W.t()                        # 一次性算完 4×8=32 个点积
print(out.shape)                       # (4, 8)
```

矩阵乘法的本质，就是**左边每一行和右边每一列做点积**。GPU 拼命优化的，就是这个。

**第三层：连 Transformer 的注意力（Attention），核心也是点积。** 大模型最性感的那个机制，
其实就是"每个词去和别的词做点积，看谁和谁最相关"——回想第 1 节：**点积衡量两根箭头多同向**：

```python
import torch
import torch.nn.functional as F

# 3 个词，每个用 4 维向量表示
Q = torch.randn(3, 4)                  # Query
K = torch.randn(3, 4)                  # Key
V = torch.randn(3, 4)                  # Value

scores = Q @ K.t()                     # 每个词和每个词做点积 = "相关度矩阵"
attn = F.softmax(scores, dim=-1)       # 归一化成权重
out = attn @ V                         # 按相关度加权求和
print("注意力输出形状:", out.shape)        # (3, 4)
```

`Q @ K.t()` 就是在算"谁和谁同向/相关"。整个 GPT，堆的就是这套
**点积 → 加权 → 再点积**。

**那为什么不只是点积？** 因为纯点积/纯矩阵乘法叠多少层都还是线性的（多个映射可以合并成一个），
表达不了曲线。所以每层后面要插一个**非线性**（ReLU 等）"折一下"，才打破线性封闭性：

```python
import torch
import torch.nn as nn

# 纯线性叠两层，等价于一层 —— 用代码证明
torch.manual_seed(0)
x = torch.randn(5, 3)
a, b = nn.Linear(3, 4), nn.Linear(4, 2)
two = b(a(x))
W = b.weight @ a.weight                          # 合并成一台机器
one = x @ W.t() + (b.weight @ a.bias + b.bias)
print(torch.allclose(two, one, atol=1e-6))       # True —— 叠了也白叠
```

所以最准确的回答是：

> **神经网络 = 大量点积（线性代数）+ 少量折弯（非线性）。**
> 点积负责"在新空间里量分量、算相关性"，折弯负责"让模型能表达曲线"。
> 大模型的"智能"，是几千亿个点积在超高维空间里协作的涌现结果——
> 单看每一步，确实平平无奇，就是点积。

---

## 🤔 疑惑点三：离散的字/像素/整数，怎么变成"连续可导"的？

这是初学者最大的困惑：梯度下降要求**对参数可导**，可现实世界全是离散的——
一个字、一个像素灰度、一个运算符 `+`，怎么"求导"？

答案分两步：**①把离散对象塞进连续空间（embedding）；②对离散的"选择"动作做软化（softmax/温度）。**

**第一步：embedding——给每个离散符号发一个可学习的连续向量。**
字本身不可导，但"代表这个字的那串浮点数"可导。

```python
import torch
import torch.nn as nn

# 词表里 10 个词，每个词用一个 4 维连续向量表示
emb = nn.Embedding(num_embeddings=10, embedding_dim=4)

ids = torch.tensor([1, 5, 9])          # 三个离散的词 id（不可导）
vecs = emb(ids)                        # 查表 -> 连续向量（可导！）
print(vecs.shape)                      # (3, 4)
print(vecs.requires_grad)              # True —— 这些浮点数能被梯度更新
```

`nn.Embedding` 本质就是**一张可学习的查找表**：离散 id 进去，连续向量出来。
训练时，梯度回传去微调这张表里的浮点数。**"猫"和"狗"这两个离散符号的语义远近，
就编码在它们对应向量的点积里**（又是点积！）。

**第二步：argmax 不可导，就用 softmax 这个"软化版"替身。**
"在 10 个词里挑概率最大的那个"是 `argmax`——一个阶梯式的硬选择，梯度处处为 0，没法学。
softmax 把它**软化成连续的概率分布**，于是可导：

```python
import torch

logits = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)

hard = logits.argmax()                 # 硬选择：返回索引 0，不可导
soft = torch.softmax(logits, dim=0)    # 软选择：连续概率 [0.66, 0.24, 0.10]
print("硬:", hard, " 软:", soft)

soft.sum().backward()                  # 软版本能求导
print("softmax 可以回传梯度:", logits.grad is not None)   # True
```

温度参数还能在"软"和"硬"之间连续滑动——温度→0 时 softmax 逼近 argmax：

```python
import torch
logits = torch.tensor([2.0, 1.0, 0.1])
for T in [3.0, 1.0, 0.1]:
    p = torch.softmax(logits / T, dim=0)
    print(f"温度 {T}: {p.numpy().round(3)}")   # T 越小越接近 one-hot 硬选择
```

**这就是"离散变连续可导"的全部魔法：**

> 用 embedding 把离散符号搬进连续向量空间（可以求导的浮点数），
> 用 softmax/温度把离散的"选择"动作软化成连续概率（可以求导的分布）。
> 训练完，再用 argmax 把连续概率"硬化"回离散的字——
> **训练时连续可导，推理时离散输出。** 两头都要，中间靠这两招缝合。

> 像素其实天生就是连续的（0–255 归一化成 0–1 的浮点），所以图像神经网络
> 一开始就不太有这个烦恼；真正难的是文字、类别、动作这类**纯符号**的东西。

---

## 6. 训练，就是把"搬空间的机器"一点点调好（线性代数遇上微积分）

前面所有矩阵都是手写死的。**训练，就是让模型自己学出这些矩阵该长什么样。**
把第 2~5 节缝进一个最小训练循环：

```python
import torch
import torch.nn as nn

# 目标：学一个 2x2 矩阵，把输入"旋转 90°"
torch.manual_seed(0)
target = torch.tensor([[0.0, -1.0],
                       [1.0,  0.0]])    # 真正的旋转矩阵（模型不知道）

x = torch.randn(200, 2)
y = x @ target.t()                      # 标准答案

model = nn.Linear(2, 2, bias=False)     # 模型：一个待学习的 2x2 矩阵
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(300):
    pred = model(x)
    loss = ((pred - y) ** 2).mean()     # 搬得对不对
    opt.zero_grad(); loss.backward(); opt.step()

print("学到的矩阵:\n", model.weight.detach().round(decimals=2))
print("目标矩阵:\n", target)             # 两者几乎一样 —— 模型"猜出"了旋转
```

跑完你会看到 `model.weight` 收敛到那个旋转矩阵。
**这就是深度学习的全部：用梯度下降，把一台台"搬空间的机器"调到正好把数据搬对位置。**
线性代数提供"机器"，微积分提供"怎么调"。

---

## 🤔 疑惑点四：GPU 是不是就是一台"大型向量机"？

**基本上，是的——而且这个直觉惊人地准确。** 但要补一个关键修正。

CPU 像一个**博士**：单核极聪明，能处理复杂分支逻辑，但一次只干一件事（核心少）。
GPU 像**几千个小学生**：每个都只会做简单算术，但**几千个同时做同一道题**。
而矩阵乘法——第 6 节我们反复在做的——恰好就是"对成千上万个数，同时做同样的乘加"。
这正是 GPU 的主场：

```python
import torch, time

# 一个 4096x4096 的矩阵乘法 = 约 1370 亿次乘加
A = torch.randn(4096, 4096)
B = torch.randn(4096, 4096)

t = time.time(); _ = A @ B; cpu_t = time.time() - t
print(f"CPU 矩阵乘法耗时: {cpu_t*1000:.1f} ms")

if torch.cuda.is_available():
    Ag, Bg = A.cuda(), B.cuda()
    torch.cuda.synchronize()
    t = time.time(); _ = Ag @ Bg; torch.cuda.synchronize()
    gpu_t = time.time() - t
    print(f"GPU 矩阵乘法耗时: {gpu_t*1000:.1f} ms")     # 通常快几十到上百倍
    print(f"加速比: {cpu_t/gpu_t:.0f}x")
```

为什么差这么多？因为矩阵乘法的每个输出元素，都是**互不依赖**的一次点积，
天然可以并行。GPU 有几千个核心，一口气把几千个点积同时算了：

```python
import torch

# 矩阵乘法 = 一大批彼此独立的点积，谁也不等谁 -> 完美并行
A = torch.randn(3, 4)
B = torch.randn(4, 5)
manual = torch.stack([torch.stack([torch.dot(A[i], B[:, j])  # 每个格子一个点积
                                   for j in range(5)])
                      for i in range(3)])
print(torch.allclose(manual, A @ B))   # True —— matmul 就是一堆并行点积
```

**需要补的那个修正：** 现代 GPU 不只是"向量机"，更准确说是"**矩阵机/张量机**"。
NVIDIA 的 **Tensor Core** 是直接为"小矩阵块乘加"定制的硬件电路，
比起一个数一个数地算向量，它一个时钟周期就吞下一整块矩阵。所以更精确的说法是：

> **GPU 是一台大规模并行的"张量计算机"。**
> 它的存在意义，就是把第 6 节那种"成千上万个互相独立的点积/矩阵乘"在同一瞬间算完。
> 深度学习之所以在 2012 年后爆发，硬件上就一句话：
> **有人发现，神经网络的计算 = 海量矩阵乘法 = 正好是 GPU 最擅长的事。**

线性代数（矩阵乘法）、神经网络（堆叠点积）、GPU（并行算点积）——
**三者是同一件事在数学、模型、硬件三个层面的三张面孔。**

---

## 7. 把所有画面缝合起来

最后回到开头那句话，现在每个概念都有了画面和代码出处：

| 线性代数概念 | 课本怎么讲（抽象） | 这篇文章怎么看（画面） | 在深度学习里是什么 |
|---|---|---|---|
| **向量** | 一列数 | 空间里一根有方向的箭头（第 1 节） | 一个样本 / 一个词的 embedding |
| **点积** | Σ 对应相乘 | 两根箭头有多同向（第 1 节） | 神经元、注意力相关度（疑惑点二） |
| **矩阵** | 数表 + 运算法则 | 把整个空间搬动的机器（第 2 节） | `nn.Linear` / 权重矩阵（第 5 节） |
| **行列式** | 按行展开 | 面积被放大几倍，0 = 拍扁不可逆（第 3 节） | 衡量映射是否丢信息 |
| **特征向量** | 解 `det(A-λI)=0` | 映射里方向不变的骨架轴（第 4 节） | PCA / 数据主方向（疑惑点一） |
| **矩阵乘法** | 行乘列 | 一堆独立点积，可并行（疑惑点四） | GPU 上的核心运算 |

三句话总结这篇文章：

> **线性代数的灵魂是"矩阵就是映射"**——一台把空间整体搬动的机器（第 2、3 节）；
> **神经网络的灵魂是"海量点积"**——在高维空间里量分量、算相关，再用非线性折弯（疑惑点二）；
> **而离散变可导（embedding + softmax）让符号也能进这套机器，GPU 让这套机器一瞬间跑完**（疑惑点三、四）。

当年线性代数没讲明白，不是因为它难，是因为没人告诉你：
**这些符号，最后全都活在一段能跑起来的 PyTorch 代码里。**
现在你把上面每段代码都跑一遍，比期末复习四个晚上都管用。
