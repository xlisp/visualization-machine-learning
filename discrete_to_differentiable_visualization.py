"""
离散怎么变成"连续可导"的？—— 可视化（3Blue1Brown 风格）
========================================================

困惑：选词、选类别、挑最大值，本质是一个"硬选择"(argmax) —— 一个台阶函数。
台阶函数处处是平的，斜率为 0，梯度下降"无坡可下"，根本学不动。
那神经网络是怎么对这种离散选择求导、做训练的？

答案：用 softmax 把"硬台阶"软化成一条**光滑的斜坡**，斜坡处处有坡度（导数≠0），
梯度就有方向可跟了。温度 T 像一个旋钮：T 大→很软的缓坡，T→0→逼近原来的硬台阶。

这个动画让你**亲眼看见**这件事：
  左：硬选择(灰台阶) vs 软选择 softmax(蓝曲线)。蓝线上画了一条切线 ——
      切线的斜率就是"梯度"。台阶没有切线(不可导)，曲线有(可导)。
  右：两者的"梯度"(导数)。台阶的梯度恒为 0(学不动)；
      softmax 的梯度是一个**非零的鼓包**(有东西可跟，能训练)。
  温度 T 从大到小来回扫：你会看到软坡一点点变陡，最终逼近那个学不动的硬台阶。

运行: python discrete_to_differentiable_visualization.py
依赖: numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# x = 两个 logit 的差 (z0 - z1)；纵轴 = 分给类别0 的概率
x = np.linspace(-6, 6, 600)


def softmax_p0(x, T):
    return 1.0 / (1.0 + np.exp(-x / T))          # 两类 softmax = sigmoid(x/T)


def softmax_grad(x, T):
    p = softmax_p0(x, T)
    return (1.0 / T) * p * (1.0 - p)             # 导数：一个鼓包


hard_p0 = (x > 0).astype(float)                  # argmax：硬台阶
X0 = 0.4                                          # 在这个点上画切线，演示"斜率=梯度"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))
FRAMES = 120


def update(frame):
    # 温度 T：1.6 -> 0.15 -> 1.6 来回扫
    s = 1 - abs(1 - 2 * frame / FRAMES)          # 0->1->0
    T = 1.6 * (1 - s) + 0.15 * s

    p = softmax_p0(x, T)
    g = softmax_grad(x, T)

    # ---------------- 左：硬台阶 vs 软曲线 + 切线 ----------------
    axL.clear()
    axL.plot(x, hard_p0, "--", color="#999999", lw=2,
             label="硬选择 argmax（台阶，不可导）")
    axL.plot(x, p, color="#1f77b4", lw=3,
             label=f"软选择 softmax，温度 T={T:0.2f}")

    # 在 X0 处画切线：切线斜率 = 该点导数 = 梯度
    y0 = softmax_p0(np.array([X0]), T)[0]
    slope = softmax_grad(np.array([X0]), T)[0]
    tx = np.array([X0 - 2.5, X0 + 2.5])
    axL.plot(tx, y0 + slope * (tx - X0), color="#d62728", lw=2)
    axL.scatter([X0], [y0], color="#d62728", s=70, zorder=5)
    axL.text(X0 + 0.2, y0 - 0.12, f"切线斜率(=梯度)={slope:0.2f}",
             color="#d62728", fontsize=11)

    axL.set_title("把'硬台阶'软化成'光滑斜坡'：曲线有切线 = 可导\n"
                  "（台阶到处是平的，没有切线 = 不可导）", fontsize=12)
    axL.set_xlabel("z0 − z1  （两个 logit 的差）")
    axL.set_ylabel("分给类别 0 的概率")
    axL.set_ylim(-0.15, 1.15)
    axL.legend(loc="upper left", fontsize=10)

    # ---------------- 右：两者的梯度（导数） ----------------
    axR.clear()
    axR.axhline(0, color="#999999", lw=2, ls="--",
                label="argmax 的梯度 ≡ 0（无坡可下，学不动）")
    axR.plot(x, g, color="#1f77b4", lw=3,
             label="softmax 的梯度（非零鼓包，可训练）")
    axR.fill_between(x, 0, g, color="#1f77b4", alpha=0.15)
    axR.scatter([X0], [slope], color="#d62728", s=70, zorder=5)
    axR.set_title("梯度对比：可导的关键就在这里\n"
                  "softmax 处处有非零斜率 → 反向传播有方向可跟", fontsize=12)
    axR.set_xlabel("z0 − z1")
    axR.set_ylabel("∂(概率) / ∂(logit 差)")
    axR.set_ylim(-0.2, 4.5)
    axR.legend(loc="upper left", fontsize=10)


anim = FuncAnimation(fig, update, frames=FRAMES, interval=60, repeat=True)

# 想存成 gif 取消下一行注释（需要 pillow）:
# anim.save("discrete_to_differentiable_visualization.gif", writer="pillow", fps=20)

plt.tight_layout()
plt.show()
