# 大学4年没讲明白的神经网络数学，被一段 PyTorch 代码讲透了

> 为什么神经网络能学习？为什么 ReLU 如此重要？为什么 MSELoss 对应高斯分布？为什么 Adam 比 SGD 更聪明？一篇文章从代码出发，讲透 AI 背后的数学本质。
>
> 本文以仓库里两个真实例子为线索：
>
> - `heuristic_thinking/calculator_nn2.py` —— 一个想学会四则运算的全连接网络
> - `heuristic_thinking/2013_nonlinear_fitting/2013_nonlinear_fitting.py` —— 一个拟合 `sin(x) + 噪声` 的 `NonlinearModel`
>
> 目标：**不写一个数学公式**，全部用可以运行的 Python/PyTorch 代码，把藏在这两段代码背后的
> **线性代数、微积分、概率** 三块知识讲清楚，并解释为什么从 `LinearModel` 一步步走到 `NonlinearModel`。

---

## 0. 一句话主线

整个机器学习训练，就是下面这个循环：

```python
for epoch in range(epochs):
    pred = model(x)            # ① 线性代数：一堆矩阵乘法 + 非线性
    loss = criterion(pred, y)  # ② 概率：用"误差多大"度量"模型有多不可信"
    optimizer.zero_grad()
    loss.backward()            # ③ 微积分：自动求每个参数对 loss 的导数
    optimizer.step()           # ④ 微积分：沿导数反方向走一小步
```

- `model(x)` 这一步，本质是**线性代数**（矩阵乘法）。
- `criterion(pred, y)` 这一步，本质是**概率**（极大似然 → 最小二乘）。
- `loss.backward()` 和 `optimizer.step()` 这两步，本质是**微积分**（链式法则 + 梯度下降）。

下面逐块拆开。

---

## 1. 起点：线性回归 `LinearModel`

仓库里的 `NonlinearModel` 用了 3 个 `nn.Linear`。要理解它，先把它砍到只剩一层 —— 那就是线性回归。

### 1.1 一个 `nn.Linear` 到底在算什么（线性代数）

`nn.Linear(in, out)` 不是魔法，它就是一次矩阵乘法加偏置。我们手写一遍，和 PyTorch 对照：

```python
import torch
import torch.nn as nn

x = torch.tensor([[2.0, 3.0]])      # 1 个样本，2 个特征  形状 (1, 2)

layer = nn.Linear(2, 1)             # 2 个输入 -> 1 个输出
W = layer.weight                    # 形状 (1, 2) —— 输出 x 输入
b = layer.bias                      # 形状 (1,)

# PyTorch 内部做的事：y = x @ W^T + b
manual = x @ W.t() + b
print(torch.allclose(layer(x), manual))   # True
```

所以一个"神经元"就是：**把输入向量和一行权重做点积，再加一个偏置**。
点积 = 加权求和 = 线性代数里最基础的运算。

`calculator_nn2.py` 里的 `nn.Linear(3, 128)`，意思就是：
3 维输入（两个数 + 运算符编码）通过一个 `128 × 3` 的矩阵，被"投影"成 128 维。
矩阵的每一行，都是一个看待输入的不同角度。

### 1.2 把"线性回归"写成最小可运行的代码

```python
import torch

# 造一批服从 y = 2x + 1 的数据，再加一点噪声
x = torch.linspace(-1, 1, 50).unsqueeze(1)      # (50, 1)
y = 2 * x + 1 + torch.randn(50, 1) * 0.1        # 真值 + 噪声

model = torch.nn.Linear(1, 1)                   # 只有 w 和 b 两个参数
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

for _ in range(500):
    pred = model(x)
    loss = loss_fn(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()

print(model.weight.item(), model.bias.item())  # 约等于 2 和 1
```

这就是 `NonlinearModel` 的"婴儿版"。它能拟合**直线**，但拟合不了 `sin(x)` —— 原因下一节揭晓。

---

## 2. 为什么线性不够：线性的"封闭性"陷阱（线性代数）

直觉上你会想：那我多堆几层 `nn.Linear` 不就更强了？错。**多个线性层叠起来，还是一个线性层。** 用代码证明：

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.randn(5, 3)

# 两层纯线性，中间不加任何非线性
a = nn.Linear(3, 4)
b = nn.Linear(4, 2)
two_layers = b(a(x))

# 把它们合并成"一层"：W = Wb @ Wa, bias = Wb @ ba + bb
W = b.weight @ a.weight
bias = b.weight @ a.bias + b.bias
one_layer = x @ W.t() + bias

print(torch.allclose(two_layers, one_layer, atol=1e-6))   # True
```

结论：`Linear(Linear(x))` 恒等于某个 `Linear(x)`。
无论叠多少层，只要中间没有非线性，模型能表达的永远只是**一条直线 / 一个超平面**。
这就是为什么 `sin(x)` 这种弯弯曲曲的曲线，纯线性网络永远拟合不了。

---

## 3. 转折点：激活函数让网络"拐弯"（线性代数 → 非线性）

看 `NonlinearModel.forward`：

```python
def forward(self, x):
    x = torch.relu(self.fc1(x))   # <-- relu 是关键
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```

`torch.relu` 就一行逻辑：**负数变 0，正数不变**。

```python
import torch
z = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0])
print(torch.relu(z))      # tensor([0., 0., 0., 1., 3.])
print(torch.maximum(z, torch.zeros_like(z)))  # 完全等价
```

它本身简单到不像数学，但正是这个"折一下"打破了第 2 节的线性封闭性。
**每个 ReLU 神经元 = 一个铰链（hinge），在某个位置把直线折弯。** 把很多段折线拼起来，就能逼近任意曲线：

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 只用一层隐藏层 + relu，看它怎么用"折线"逼近 sin
x = torch.linspace(-6, 6, 200).unsqueeze(1)
target = torch.sin(x)

net = nn.Sequential(nn.Linear(1, 30), nn.ReLU(), nn.Linear(30, 1))
opt = torch.optim.Adam(net.parameters(), lr=0.01)

for _ in range(2000):
    loss = ((net(x) - target) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

plt.plot(x, target, label="sin(x)")
plt.plot(x, net(x).detach(), label="ReLU 折线逼近")
plt.legend(); plt.show()   # 你会看到一条由许多小直线段拼出来的"假 sin"
```

这就是 **通用逼近定理（Universal Approximation）** 的代码直觉：
*线性层负责"在哪折、折多大"，激活函数负责"真的折下去"。*
这正是从 `LinearModel` 到 `NonlinearModel` 的本质飞跃 —— 仓库里那张
`2013_nonlinear_fitting_pytorch.png` 就是它拟合成功的结果。

---

## 4. 损失函数从哪来：MSE 背后是概率（概率论）

两份代码都用了 `nn.MSELoss()`（均方误差）。为什么是"平方"，不是"绝对值"或"四次方"？
答案来自概率：**MSE 是"假设噪声服从高斯分布"时的极大似然估计。**

### 4.1 先看噪声怎么造出来的

回看 `2013_nonlinear_fitting.py` 的数据：

```python
y = torch.sin(x) + torch.rand(n, 1) * 0.5
```

数据 = **一个确定的规律（sin）** + **一个随机的噪声**。
真实世界的测量永远带噪声，所以 `y` 不是一个定值，而是一个**随机变量**。
模型要学的，是这个随机变量的**期望（均值）**。

### 4.2 用代码说明"为什么平方误差 = 高斯假设"

假设噪声是高斯的，那么"模型预测得有多准"可以用高斯概率密度来打分。
我们把"最小化 MSE"和"最大化高斯似然"放在一起跑，看它们是不是同一回事：

```python
import torch

pred = torch.tensor([1.0, 2.0, 3.0])
true = torch.tensor([1.2, 1.9, 3.5])
err = pred - true

# 方案 A：均方误差
mse = (err ** 2).mean()

# 方案 B：假设误差 ~ 高斯(0, sigma)，写出"负对数似然"
sigma = 1.0
nll = (err ** 2 / (2 * sigma ** 2)).mean()   # 高斯 NLL 去掉常数项后剩下的部分

print(mse, nll * 2)   # 成正比 —— 最小化 MSE 等价于最大化高斯似然
```

一句话：**你选 MSE，就等于悄悄声明"我认为数据噪声是高斯的"。**
如果噪声有很多离群点（重尾分布），那 MSE 就不再是最优选择，这时会换成 L1（绝对值）损失 —— 那对应的是拉普拉斯分布假设。损失函数的选择，本质是对噪声分布的概率假设。

### 4.3 这也解释了 `calculator_nn2.py` 里的"loss 太大"现象

文件注释里记录了三次运行：

```
run1: Loss: 2184328  ->  32*3 预测成 296（离谱）
run2: Loss: 764      ->  389+88 预测成 -964（还是错）
run3: Loss: 22       ->  32*3 预测成 94.7（接近 96 了）
```

为什么 loss 数字这么吓人？因为 MSE 是**平方**：乘法结果能到 100×100=10000，
误差平方一下就到了百万级。这正是平方的双刃剑 ——
对大误差极其敏感（高斯假设下大偏差概率极低，所以惩罚极重）。
这也提示了一个真实工程问题：**没做归一化（normalization）**，
输入范围 1~100、输出范围却能到 10000，量纲差异让训练极难收敛。

---

## 5. 训练的发动机：梯度下降（微积分）

现在到了 `loss.backward()` 和 `optimizer.step()`。这两行是整篇文章里**微积分**含量最高的地方。

### 5.1 导数 = "我往哪个方向动，loss 会下降"

导数的定义就是"输入动一点点，输出动多少"。我们不用公式，用**数值差分**亲手算一次导数，
再让 PyTorch 用 `backward()` 算一次，对比两者：

```python
import torch

def f(w):
    return (w - 3) ** 2          # 一个最小值在 w=3 的碗形函数

# 方法一：手动数值求导（微积分的原始定义：差商）
w = 5.0
eps = 1e-4
numeric_grad = (f(torch.tensor(w + eps)) - f(torch.tensor(w - eps))) / (2 * eps)

# 方法二：PyTorch 自动求导
wt = torch.tensor(w, requires_grad=True)
f(wt).backward()
auto_grad = wt.grad

print(numeric_grad, auto_grad)   # 两者都约等于 4.0
```

两种方法结果一致。`backward()` 不是黑魔法，它就是在自动、精确地做你手动做的事。

### 5.2 梯度下降：顺着导数反方向走

知道了导数（上坡方向），把参数往**反方向**挪一点，loss 就会降。
手写一个梯度下降，亲眼看着它滚到谷底：

```python
import torch

w = torch.tensor(5.0, requires_grad=True)
lr = 0.1                          # 学习率：每步走多大

for step in range(20):
    loss = (w - 3) ** 2
    loss.backward()               # 算出 dloss/dw
    with torch.no_grad():
        w -= lr * w.grad          # 顺着负梯度方向走一步 <-- optimizer.step() 的核心
    w.grad.zero_()                # 清空梯度 <-- optimizer.zero_grad() 的作用
    print(f"step {step}: w={w.item():.4f}")
# w 会从 5 一路逼近 3
```

把这段代码和训练循环对照：

| 手写代码 | PyTorch 等价写法 |
|---|---|
| `loss.backward()` | `loss.backward()` |
| `w -= lr * w.grad` | `optimizer.step()` |
| `w.grad.zero_()` | `optimizer.zero_grad()` |

**`optimizer.zero_grad()` 为什么不能漏？** 因为 PyTorch 的梯度是**累加**的。
不清零，这一轮的梯度会叠加上一轮的，方向就乱了。用代码验证：

```python
import torch
w = torch.tensor(2.0, requires_grad=True)

(w ** 2).backward(); print(w.grad)   # 4.0
(w ** 2).backward(); print(w.grad)   # 8.0 —— 没清零，累加了！
```

### 5.3 链式法则：多层网络怎么把梯度传回去

`NonlinearModel` 有三层，`fc1` 的参数离 loss 很"远"。怎么知道改 `fc1` 一点点，
loss 会变多少？答案是微积分的**链式法则**：一层层把导数乘回去。
用一个两层的小例子手动复现 PyTorch 的反向传播：

```python
import torch

x = torch.tensor(1.5)
w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(3.0, requires_grad=True)

# 前向：h = w1 * x ;  out = w2 * h ;  loss = out^2
h = w1 * x
out = w2 * h
loss = out ** 2
loss.backward()

# 手动用链式法则验证 dloss/dw1：
#   dloss/dout = 2*out
#   dout/dh    = w2
#   dh/dw1     = x
manual_grad_w1 = (2 * out) * w2 * x
print(w1.grad, manual_grad_w1.item())   # 两者相等
```

`backward()` 做的，就是从 loss 出发，沿着计算图把这些局部导数**自动地连乘**回每个参数。
这就是为什么深层的 `NonlinearModel` 也能被训练 —— 链式法则保证梯度能一路传到 `fc1`。

---

## 6. 优化器为什么用 Adam 而不是 SGD（微积分的工程进化）

两份代码都用了 `torch.optim.Adam`，而第 5 节我们手写的是最朴素的 SGD。
区别在于：Adam 会**自适应地调整每个参数的步长**，并利用历史梯度做"惯性"（动量）。
用代码感受它们的差异：

```python
import torch

def run(opt_name):
    w = torch.tensor([5.0, 5.0], requires_grad=True)
    opt = (torch.optim.SGD([w], lr=0.1) if opt_name == "SGD"
           else torch.optim.Adam([w], lr=0.1))
    for _ in range(50):
        # 一个在两个方向上"陡峭程度"差很多的椭圆碗
        loss = (w[0] - 3) ** 2 + 20 * (w[1] - 1) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach()

print("SGD :", run("SGD"))
print("Adam:", run("Adam"))   # Adam 通常更快更稳地同时收敛两个方向
```

朴素 SGD 在"地形陡峭程度不均"时容易震荡或走得慢，Adam 通过记录梯度的一阶/二阶滑动平均，
自动给每个参数一个合适的步长。这也是 `2013_nonlinear_fitting.py` 用 `lr=0.01` 的 Adam
就能在几千步内把 `sin` 拟合好的原因。

---

## 7. 概率的另一面：过拟合（probability / 泛化）

`2013_nonlinear_fitting.py` 文件末尾记录了一个重要现象：

```
1000 epochs -> Loss 0.2315   拟合刚刚好  (2013_nonlinear_fitting_pytorch.png)
5000 epochs -> Loss 0.0124   过拟合了    (2013_nonlinear_fitting_overfitting.png)
```

loss 更低，效果反而更差，这看起来矛盾，其实是概率/统计的核心议题。
回忆数据是 `y = sin(x) + 噪声`：

```python
y = torch.sin(x) + torch.rand(n, 1) * 0.5   # 信号 + 噪声
```

- 训练 1000 步：模型学到了 **`sin(x)` 这个真实信号**（泛化好）。
- 训练 5000 步：模型把 **随机噪声 `torch.rand` 也背下来了**（泛化差）。

模型没法区分"哪部分是规律、哪部分是噪声"，训练太久就会去拟合那些本应被忽略的随机抖动。
用代码量化这件事 —— 真正该关心的是**没见过的数据**上的误差：

```python
import torch, torch.nn as nn

torch.manual_seed(0)
x = torch.linspace(1, 10, 100).unsqueeze(1)
y = torch.sin(x) + torch.rand(100, 1) * 0.5

# 分成训练集和测试集 —— 测试集模型训练时从没见过
idx = torch.randperm(100)
tr, te = idx[:70], idx[70:]

net = nn.Sequential(nn.Linear(1, 10), nn.ReLU(),
                    nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))
opt = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(5000):
    loss = ((net(x[tr]) - y[tr]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if (epoch + 1) % 1000 == 0:
        with torch.no_grad():
            test_loss = ((net(x[te]) - y[te]) ** 2).mean()
        print(f"epoch {epoch+1}: 训练 loss={loss.item():.4f}  测试 loss={test_loss.item():.4f}")
# 你会看到：训练 loss 一直降，但测试 loss 降到某点后开始回升 —— 那个拐点就是该停手的地方
```

这就是 **偏差-方差权衡（bias-variance tradeoff）** 的实战版：
模型容量太小欠拟合（偏差大），训练太久/容量太大过拟合（方差大）。
应对手段（early stopping、正则化、加数据）本质都是在用概率假设约束模型，
让它只学"可复现的规律"而不是"这一批数据的偶然噪声"。

---

## 8. 把三块知识缝合回训练循环

最后回到开头那段循环，现在每一行都有了出处：

```python
for epoch in range(epochs):
    outputs = model(x)              # 线性代数：matmul 堆叠 + ReLU 折弯（第 1、3 节）
    loss = criterion(outputs, y)    # 概率论：高斯噪声假设 → MSE（第 4 节）
    optimizer.zero_grad()           # 微积分：清空累加的梯度（第 5.2 节）
    loss.backward()                 # 微积分：链式法则自动求导（第 5.1、5.3 节）
    optimizer.step()                # 微积分：沿负梯度更新（第 5.2、6 节）
```

| 知识领域 | 在代码里的位置 | 核心直觉 | 深入章节 |
|---|---|---|---|
| **线性代数** | `nn.Linear` / `model(x)` | **矩阵就是映射**：把空间整体搬动；神经元 = 投影 = 加权点积 | 第 9 节 |
| **非线性** | `torch.relu` | 折线拼曲线，打破线性封闭性，是 Nonlinear 的灵魂 | 第 3、9.6 节 |
| **概率论** | `nn.MSELoss` / 数据里的噪声 | **概率就是面积**：曲线下方那块区域；损失 = 对噪声分布的假设 | 第 10 节 |
| **微积分** | `.backward()` / `optimizer.step()` | 导数指方向，梯度下降走下坡，链式法则传到深层 | 第 5、6 节 |

两个最该记住的"看见数学"的画面：

- **矩阵就是映射** —— 一个矩阵是一台把整个空间搬动/拉斜/升维的机器，行列式是它对**面积**的缩放倍数，训练就是在调这台机器把数据搬到好切开的地方（第 9 节）。
- **概率就是面积** —— 概率是密度曲线下方那块区域的大小，总面积恒为 1，期望是这块面积的重心，MSE 则是"高斯面积最大"的等价物（第 10 节）。

### 从 `LinearModel` 到 `NonlinearModel` 的一句话总结

> 线性回归给了你一把直尺，只能画直线（第 1、2 节）；
> 加上 ReLU 这个"折弯"动作，直尺变成了能拼任意曲线的折叠尺（第 3 节）；
> 用高斯假设导出的 MSE 告诉你"拟合得有多好"（第 4 节）；
> 用微积分的梯度下降一步步把尺子掰到正确的形状（第 5、6 节）；
> 而知道**什么时候停手**，才是概率/泛化教给你的最后一课（第 7 节）。

---

## 9. 深入线性代数：矩阵就是映射

> 这一节专门把第 1、2 节里"`nn.Linear` 就是矩阵乘法"这句话，升级成一个能让你**在脑子里看见**的画面：
> **一个矩阵 = 一台把空间整体搬动的机器。** 你喂给它一个向量（一个点），它吐出另一个向量（另一个点）。
> 训练神经网络，本质就是在调这台机器，让它把数据"搬"到容易处理的位置。

### 9.1 矩阵不是数表，是动作

别把矩阵看成"一堆数字"，把它看成"对所有点的同一个操作"。
喂进去几个点，看它们被搬到哪：

```python
import torch

M = torch.tensor([[2.0, 0.0],
                  [0.0, 3.0]])     # 这台机器：横向拉 2 倍，纵向拉 3 倍

points = torch.tensor([[1.0, 0.0],   # 右
                       [0.0, 1.0],   # 上
                       [1.0, 1.0]])  # 右上角
print(points @ M.t())
# [[2, 0], [0, 3], [2, 3]]  —— 每个点都被同一条规则搬动了
```

换一个矩阵，就是换一个动作。下面四台机器分别是：旋转、剪切、投影、压扁：

```python
import torch, math

theta = math.radians(30)
rotate = torch.tensor([[math.cos(theta), -math.sin(theta)],   # 把整个平面转 30°
                       [math.sin(theta),  math.cos(theta)]])

shear  = torch.tensor([[1.0, 1.0],     # 剪切：越往上的点越往右滑（像推倒一摞书）
                       [0.0, 1.0]])

project = torch.tensor([[1.0, 0.0],    # 投影：把所有点拍扁到 x 轴上
                        [0.0, 0.0]])

v = torch.tensor([1.0, 2.0])
print(rotate @ v)    # 转了个角度，长度不变
print(shear  @ v)    # x 被 y 拖着走
print(project @ v)   # y 直接变 0，二维信息塌成一维
```

记住这个分类，下面所有概念都挂在它上面。

### 9.2 行列式 = 面积的缩放倍数（矩阵与"面积"的第一次相遇）

这里就接上了你说的"概率是面积"——**线性代数里也有面积，而且它有名字：行列式。**
取单位正方形（面积 1）的四个角，用矩阵搬动它们，搬完后的面积，正好等于行列式的绝对值：

```python
import torch

M = torch.tensor([[2.0, 1.0],
                  [0.0, 3.0]])

# 单位正方形的两条边向量
e1 = torch.tensor([1.0, 0.0])
e2 = torch.tensor([0.0, 1.0])

# 被 M 搬动后的两条边
a = M @ e1
b = M @ e2

# 平行四边形面积 = 叉积大小 = |ax*by - ay*bx|
area = abs(a[0] * b[1] - a[1] * b[0])
print("搬动后面积:", area.item())          # 6.0
print("行列式:    ", torch.det(M).item())  # 6.0  —— 完全一样
```

直觉：**行列式告诉你这台机器把面积放大/缩小了几倍。**

- `det = 2`：面积变 2 倍。
- `det = 1`：面积不变（比如旋转，它只转不拉）。
- `det = 0`：**面积被压成 0** —— 整个平面被拍扁成一条线，信息丢了，再也回不来（不可逆）。

回看 9.1 的 `project` 矩阵：

```python
import torch
project = torch.tensor([[1.0, 0.0],
                        [0.0, 0.0]])
print(torch.det(project).item())   # 0.0  —— 拍扁了，行列式为 0
```

`det = 0` 的矩阵不可逆，因为你没法从"拍扁后的影子"反推回原来的立体形状。
这就是第 2 节"多层纯线性会塌缩"的几何版本：连续的拍扁，只会更扁。

### 9.3 把网格画出来，亲眼看"映射"

光看数字不过瘾。把一张网格喂给矩阵，看整个空间怎么被扭曲：

```python
import torch
import matplotlib.pyplot as plt

# 造一张网格（一堆点）
xs, ys = torch.meshgrid(torch.linspace(-2, 2, 11),
                        torch.linspace(-2, 2, 11), indexing="xy")
grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)   # (N, 2)

M = torch.tensor([[1.0, 0.8],
                  [0.3, 1.2]])
moved = grid @ M.t()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(grid[:, 0],  grid[:, 1]);  ax[0].set_title("原始空间")
ax[1].scatter(moved[:, 0], moved[:, 1]); ax[1].set_title("被矩阵 M 映射后")
plt.show()
# 原本横平竖直的网格，被整体"斜拉"成了平行四边形网格
```

**这就是"矩阵就是映射"最直观的画面：直线还是直线、原点还是原点，但整个空间被均匀地拉斜了。**

### 9.4 特征向量：映射中"不改方向"的那几根轴

大多数点被矩阵搬动后，方向会变。但总有那么几个特殊方向，搬动后**方向不变，只是被拉长或缩短** ——
这就是特征向量，拉伸的倍数就是特征值。用代码找出来并验证：

```python
import torch

M = torch.tensor([[2.0, 1.0],
                  [1.0, 2.0]])
vals, vecs = torch.linalg.eig(M)
print("特征值:", vals.real)              # [1., 3.]

v = vecs[:, 1].real                      # 取一个特征向量
print("M @ v :", M @ v)                  # 等于 3 * v
print("3 * v :", 3 * v)                  # 方向完全一样，只是长度 ×3
```

直觉：**特征向量是这台映射机器的"骨架方向"。** 任何复杂的线性变换，
都可以理解成"沿着这几根骨架方向，各自拉伸不同倍数"。PCA 降维、协方差分析全靠它。

### 9.5 回到神经网络：`nn.Linear(3, 128)` 是一次"升维映射"

现在把这套直觉用到 `calculator_nn2.py` 的第一层 `nn.Linear(3, 128)`：

```python
import torch.nn as nn
layer = nn.Linear(3, 128)
print(layer.weight.shape)   # torch.Size([128, 3])
```

- 它是一个 `128 × 3` 的矩阵，把**3 维输入映射到 128 维空间**。
- 矩阵的**每一行**，是一个 3 维方向；输入和这一行做点积，就是把输入**投影到这个方向上**，
  量出"输入在这个方向上有多少分量"。128 行 = 从 128 个不同角度同时打量同一个输入。

为什么要升到 128 维？因为**在低维挤在一起、纠缠不清的数据，升到高维往往就能被一刀切开。**
用一个经典例子（同心圆，在 2 维无法用直线分开）演示升维的威力：

```python
import torch

# 内圈和外圈，在 2D 平面上无法用一条直线分开
theta = torch.linspace(0, 2 * 3.14159, 100)
inner = torch.stack([torch.cos(theta), torch.sin(theta)], 1) * 1.0
outer = torch.stack([torch.cos(theta), torch.sin(theta)], 1) * 2.0

# 升维：加一个 z = x^2 + y^2 维度（半径平方）
def lift(p):
    z = (p ** 2).sum(1, keepdim=True)
    return torch.cat([p, z], 1)

print("内圈的 z 值约:", lift(inner)[:, 2].mean().item())   # ≈ 1
print("外圈的 z 值约:", lift(outer)[:, 2].mean().item())   # ≈ 4
# 升到 3D 后，内外圈在 z 轴上彻底分开 —— 一个水平面 z=2.5 就能切开它们
```

`nn.Linear` 升维 + ReLU 折弯（第 3 节），合起来干的就是这件事：
**把搅在一起的数据，映射到一个能被简单切开的新空间。** 训练，就是在学"该往哪个高维空间搬"。

### 9.6 矩阵乘法 = 映射的接力（第 2 节的几何解释）

第 2 节我们用代数证明了"两层线性 = 一层线性"。现在用映射的语言重说一遍，秒懂：

```python
import torch
A = torch.tensor([[0., -1.], [1., 0.]])    # 机器 A：旋转 90°
B = torch.tensor([[2., 0.],  [0., 2.]])    # 机器 B：放大 2 倍

v = torch.tensor([1.0, 0.0])

# 先过 A 再过 B
print(B @ (A @ v))
# 等价于：先把两台机器合成一台 C = B @ A，再过一次
C = B @ A
print(C @ v)        # 结果相同
```

**矩阵乘法就是"把两个动作合并成一个动作"。** 旋转后放大，等于一台"又转又放大"的机器。
所以无论串多少台线性机器，最后都能合并成一台 —— 还是线性的。
要打破这一点，必须在中间插入一个**非线性的折弯动作**（ReLU），让"接力"没法被简单合并。
这就是为什么 `NonlinearModel` 的每个 `Linear` 后面都得跟一个 `relu`。

---

## 10. 深入概率论：概率就是面积

> 这一节把第 4、7 节里的概率，落到你说的那句话上：**概率 = 面积。**
> 一旦你把"概率"看成"曲线下方那块区域的大小"，高斯分布、最小二乘、过拟合就全部串起来了。

### 10.1 核心直觉：高度不是概率，面积才是

概率密度函数（PDF）画出来是一条曲线。曲线的**高度**不是概率，曲线下方某段的**面积**才是概率。
用最笨但最直观的办法——把面积切成很多小长方形加起来（这就是积分的定义）——验证整条高斯曲线下面积是 1：

```python
import torch, math

def gaussian_pdf(x, mu=0.0, sigma=1.0):       # 高斯密度，就是那条钟形曲线的"高度"
    return torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))

x = torch.linspace(-10, 10, 100000)
dx = x[1] - x[0]
total_area = (gaussian_pdf(x) * dx).sum()     # 把无数小长方形(高 × 宽)加起来 = 面积
print("钟形曲线下总面积:", total_area.item())  # ≈ 1.0
```

**总面积 = 1，意思是"所有可能的结果加起来，必然发生其中之一"。** 这就是"概率归一化"的几何含义。

### 10.2 "落在某个区间的概率" = 那一段的面积

问"测量值落在 -1 到 1 之间的概率是多少"，就是问"曲线在 -1 到 1 之间围出多大面积"：

```python
import torch, math

def gaussian_pdf(x, mu=0.0, sigma=1.0):
    return torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))

x = torch.linspace(-1, 1, 100000)             # 只看 [-1, 1] 这一段
dx = x[1] - x[0]
area = (gaussian_pdf(x) * dx).sum()
print("落在 [-1,1] 的概率:", area.item())      # ≈ 0.68  —— 著名的"1 个标准差 68%"
```

那个有名的"68–95–99.7 法则"，本质就是钟形曲线下三段面积的大小，没有任何神秘。

### 10.3 用"扔飞镖"数面积：蒙特卡洛

如果曲线复杂到积不动，还有一招更直白的"概率就是面积"：**往一个方框里随机扔点，
落在曲线下方的点的比例，就是面积占比。** 用它估计圆周率 π（圆的面积）：

```python
import torch

n = 1_000_000
pts = torch.rand(n, 2)                       # 在 1×1 方框里随机撒 100 万个点
inside = (pts[:, 0] ** 2 + pts[:, 1] ** 2) <= 1.0   # 落在 1/4 圆内的点
ratio = inside.float().mean()                # 落在圆内的比例 = 面积比例
print("π 的估计值:", (ratio * 4).item())      # ≈ 3.14
```

这段代码把"概率"和"面积"彻底画上等号：**概率 = 有利结果的点数 / 全部点数 = 有利区域面积 / 全部面积。**
深度学习里的随机采样、Dropout、数据增强、强化学习的采样，骨子里都是这个"扔飞镖估面积"的思想。

### 10.4 期望 = 面积的"重心"

第 1 节说"模型要学的是噪声数据的均值/期望"。期望是什么？
是**用概率（面积）给每个取值加权后的平均位置**，也就是整块面积的重心。用代码算：

```python
import torch

# 一个有偏的骰子：6 点出现概率更高
values = torch.tensor([1., 2., 3., 4., 5., 6.])
probs  = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])   # 面积分配，加起来=1
print("概率和:", probs.sum().item())                     # 1.0
print("期望(重心):", (values * probs).sum().item())      # 4.6，被 6 拉过去了

# 验证：真去掷 100 万次，平均值应该逼近这个期望
samples = torch.multinomial(probs, 1_000_000, replacement=True) + 1
print("实测平均:", samples.float().mean().item())        # ≈ 4.6
```

**期望 = Σ(取值 × 它那块面积)。** 神经网络的输出，学的就是这个"重心位置"；
MSE 损失之所以合理，正因为让 MSE 最小的预测值，恰好就是数据的期望（重心）。

### 10.5 回到 MSE：最大化"面积/高度"如何变成最小化平方误差

第 4 节给了结论，这里用代码补全推理链。
"模型有多可信" = "在模型眼里，真实数据出现的概率有多大" = 真实点对应的那些**密度高度的乘积**（似然）。
我们让模型去**最大化这个乘积**，看它是不是真的等价于最小化平方误差：

```python
import torch, math

torch.manual_seed(0)
# 真实数据：围绕某个均值波动（高斯噪声）
data = 3.0 + torch.randn(2000) * 1.0

def neg_log_likelihood(mu):
    # 每个数据点的高斯密度"高度"，取对数再求和（对数把乘积变求和，数值更稳）
    sigma = 1.0
    log_height = -(data - mu) ** 2 / (2 * sigma ** 2) - math.log(sigma * math.sqrt(2 * math.pi))
    return -log_height.sum()        # 负号：最大化似然 = 最小化负对数似然

# 扫一遍不同的 mu，看哪个 mu 让数据最"可信"
mus = torch.linspace(0, 6, 601)
nll = torch.stack([neg_log_likelihood(m) for m in mus])
best_mu = mus[nll.argmin()]
print("最大似然估计的 mu:", best_mu.item())   # ≈ 3.0
print("数据的平均值:     ", data.mean().item()) # ≈ 3.0  —— 两者一致！
```

结论用代码自己证明了：**"让真实数据出现的概率（密度高度乘积）最大"** 和 **"让平方误差最小"** 是同一件事，
而它们的答案都是**数据的均值**。这就是为什么两份代码都理直气壮地用 `nn.MSELoss()`。

### 10.6 数据里的噪声，也是一块面积

`2013_nonlinear_fitting.py` 的噪声是这么造的：

```python
y = torch.sin(x) + torch.rand(n, 1) * 0.5
```

`torch.rand` 是**均匀分布**：它的密度曲线是一个矩形。"均匀"意味着这块面积被平摊在 [0, 0.5) 上，
每个值出现的概率都一样。验证它就是个面积为 1 的矩形：

```python
import torch

u = torch.rand(1_000_000) * 0.5             # 均匀分布在 [0, 0.5)
print("落在前一半 [0,0.25) 的比例:", (u < 0.25).float().mean().item())  # ≈ 0.5
print("均值(矩形重心):", u.mean().item())    # ≈ 0.25，正好在区间中点
# 矩形面积 = 宽(0.5) × 高(1/0.5=2) = 1，依然满足"总概率=1"
```

理解了噪声的分布形状，就理解了第 7 节的过拟合：
模型训练太久，会去拟合这块**本该被当成随机面积忽略**的均匀噪声，
把"每个点偶然偏移了多少"也死记硬背下来，于是失去了泛化能力。

### 10.7 分类问题里的 softmax：把任意分数变成一组"面积"

`calculator_nn2.py` 把运算符编码成 0/1/2/3 当输入。如果反过来要让网络**输出**"这是哪种运算"
（分类问题），就需要把网络吐出的任意实数分数，变成一组**加起来等于 1 的概率**——这正是 softmax 干的活，
它就是在强行制造"总面积 = 1"：

```python
import torch

scores = torch.tensor([2.0, 1.0, 0.1, -1.0])   # 网络对 4 种运算打的原始分
probs = torch.softmax(scores, dim=0)
print(probs)                # 每个都在 0~1 之间
print("概率之和:", probs.sum().item())   # 1.0  —— 又一次"总面积归一化"
```

无论是回归用的高斯密度，还是分类用的 softmax，背后是同一条铁律：
**概率必须是一块面积，而所有可能性的面积加起来必须等于 1。**

---

## 附：两个例子的对照

| | `calculator_nn2.py` | `2013_nonlinear_fitting.py` |
|---|---|---|
| 网络 | `Linear(3,128)→Linear(128,64)→Linear(64,1)` + ReLU | `Linear(1,10)→Linear(10,10)→Linear(10,1)` + ReLU |
| 输入 | 两个数 + 运算符编码 (3维) | 一个标量 x (1维) |
| 目标 | 学会 +、-、×、÷ | 拟合 `sin(x)+噪声` |
| 损失 | `MSELoss` | `MSELoss` |
| 优化器 | `Adam(lr=0.001)` | `Adam(lr=0.01)` |
| 暴露的真实问题 | 没做归一化 → loss 百万级、难收敛（第 4.3 节） | 训练太久 → 过拟合（第 7 节） |

两个例子用的是**完全相同的数学骨架**，只是输入输出维度和数据规律不同。
理解了这套骨架，你就理解了几乎所有前馈神经网络。
