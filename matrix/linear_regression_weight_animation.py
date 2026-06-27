"""
亲眼看着"权重"被改：线性回归训练过程的动态可视化
==================================================

最简单的神经网络就两个参数：斜率 w 和截距 b（y = w*x + b）。
训练就是不停地微调这两个数，让直线越来越贴合数据。

这个脚本把训练过程"放慢镜头"，三个面板同步播放：
  左   —— 拟合直线怎么一点点转到正确角度；
  中   —— **权重网格**：把 w、b 以及它们的梯度直接以数字写在格子里，
          每一帧都看见这几个数在被改写（这就是"权重"最朴素的样子）；
  右   —— 在 loss 等高线图上看参数点 (w, b) 怎么一步步滚到谷底。
**这就是 optimizer.step() 每一步在真实地修改权重的样子。**

运行: python linear_regression_weight_animation.py
依赖: torch numpy matplotlib
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(0)

# ---------------------------------------------------------------
# 1. 造一批服从 y = 2x + 1 的带噪声数据（真实权重 w=2, b=1）
# ---------------------------------------------------------------
x = torch.linspace(-3, 3, 60).unsqueeze(1)
y = 2 * x + 1 + torch.randn(60, 1) * 0.7

# ---------------------------------------------------------------
# 2. 故意从一个"很离谱"的起点开始：w=0, b=0
# ---------------------------------------------------------------
w = torch.tensor([[0.0]], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
lr = 0.03

# 记录每一步的 (w, b)，用来在等高线上画轨迹
traj = []


grads = {"w": 0.0, "b": 0.0}             # 记录最近一次的梯度，给"权重网格"展示用


def train_step():
    global w, b                          # w/b 在函数里被 -= 修改，必须声明为全局
    pred = x @ w + b
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    grads["w"], grads["b"] = w.grad.item(), b.grad.item()   # 清零前先存下来
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_(); b.grad.zero_()
    return loss.item()


# ---------------------------------------------------------------
# 3. 预先算好 loss 等高线（loss 随 w, b 变化的"地形图"）
# ---------------------------------------------------------------
ws = np.linspace(-1, 5, 120)
bs = np.linspace(-3, 5, 120)
WW, BB = np.meshgrid(ws, bs)
xn, yn = x.numpy().ravel(), y.numpy().ravel()
# 对每个 (w,b) 组合算一次均方误差
LOSS = np.zeros_like(WW)
for i in range(WW.shape[0]):
    for j in range(WW.shape[1]):
        pred = WW[i, j] * xn + BB[i, j]
        LOSS[i, j] = np.mean((pred - yn) ** 2)

# ---------------------------------------------------------------
# 4. 三面板动画
# ---------------------------------------------------------------
fig, (axL, axG, axR) = plt.subplots(1, 3, figsize=(17, 5.5))
FRAMES = 120


def draw_weight_grid(ax, cw, cb, loss, frame):
    """把权重 w、b 和它们的梯度，以数字直接写进网格里。"""
    ax.clear()
    cells = np.array([[cw, cb],                 # 第一行：当前权重值
                      [grads["w"], grads["b"]]])  # 第二行：本步梯度
    ax.imshow(cells, cmap="coolwarm", vmin=-3, vmax=3, aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cells[i, j]:+.3f}", ha="center", va="center",
                    fontsize=20, fontweight="bold", color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["w (斜率)", "b (截距)"], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["权重值", "梯度 ∂loss/∂"], fontsize=11)
    ax.set_title(f"权重网格（数字在被改写）  step={frame}\n"
                 f"更新规则: 新权重 = 旧权重 − lr × 梯度", fontsize=11)


def update(frame):
    loss = train_step()
    cw, cb = w.item(), b.item()
    traj.append((cw, cb))

    # ---- 左：数据 + 当前拟合直线 ----
    axL.clear()
    axL.scatter(x.numpy(), y.numpy(), s=14, c="#1f77b4", label="数据")
    xs = np.linspace(-3, 3, 10)
    axL.plot(xs, cw * xs + cb, c="#d62728", lw=2.5,
             label=f"y = {cw:.2f}·x + {cb:.2f}")
    axL.plot(xs, 2 * xs + 1, "--", c="#2ca02c", lw=1, label="真实 y = 2x + 1")
    axL.set_title(f"拟合直线在转动   loss = {loss:.3f}")
    axL.set_xlim(-3.2, 3.2); axL.set_ylim(-8, 9)
    axL.legend(loc="upper left", fontsize=9)

    # ---- 中：权重网格（数字） ----
    draw_weight_grid(axG, cw, cb, loss, frame)

    # ---- 右：loss 地形图 + 参数点的下山轨迹 ----
    axR.clear()
    cs = axR.contourf(WW, BB, LOSS, levels=30, cmap="viridis")
    tr = np.array(traj)
    axR.plot(tr[:, 0], tr[:, 1], "-o", c="white", ms=3, lw=1.2,
             label="参数 (w, b) 的轨迹")
    axR.scatter([cw], [cb], c="red", s=80, zorder=5, label="当前权重")
    axR.scatter([2], [1], marker="*", c="yellow", s=200, zorder=5,
                edgecolors="k", label="真正的谷底 (2, 1)")
    axR.set_title(f"权重在 loss 地形上下山   w={cw:.2f}  b={cb:.2f}")
    axR.set_xlabel("w (斜率)"); axR.set_ylabel("b (截距)")
    axR.legend(loc="upper right", fontsize=8)


anim = FuncAnimation(fig, update, frames=FRAMES, interval=60, repeat=False)

# 想存成 gif 取消下一行注释（需要 pillow）:
# anim.save("linear_regression_weight_animation.gif", writer="pillow", fps=15)

plt.tight_layout()
plt.show()
