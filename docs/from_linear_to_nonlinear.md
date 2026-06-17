# 从线性回归到非线性拟合：用代码讲透背后的数学

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

| 知识领域 | 在代码里的位置 | 核心直觉 |
|---|---|---|
| **线性代数** | `nn.Linear` / `model(x)` | 神经元 = 加权点积；多层线性会塌缩成一层 |
| **非线性** | `torch.relu` | 折线拼曲线，打破线性封闭性，是 Nonlinear 的灵魂 |
| **概率论** | `nn.MSELoss` / 数据里的噪声 | 损失函数 = 对噪声分布的假设；过拟合 = 学了噪声 |
| **微积分** | `.backward()` / `optimizer.step()` | 导数指方向，梯度下降走下坡，链式法则传到深层 |

### 从 `LinearModel` 到 `NonlinearModel` 的一句话总结

> 线性回归给了你一把直尺，只能画直线（第 1、2 节）；
> 加上 ReLU 这个"折弯"动作，直尺变成了能拼任意曲线的折叠尺（第 3 节）；
> 用高斯假设导出的 MSE 告诉你"拟合得有多好"（第 4 节）；
> 用微积分的梯度下降一步步把尺子掰到正确的形状（第 5、6 节）；
> 而知道**什么时候停手**，才是概率/泛化教给你的最后一课（第 7 节）。

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
