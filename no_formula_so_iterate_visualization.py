"""
没有求根公式，就一步步逼近 —— 牛顿法 = 梯度下降的祖宗
======================================================

一句话要让你"看见"的事：
    五次以上的方程【没有求根公式】(阿贝尔-鲁菲尼定理)。
    解不出公式怎么办？——【不求公式，只求一步步靠近】。
    这个"沿着切线往前挪一点、再挪一点"的迭代思想，
    正是深度学习里梯度下降(optimizer.step)的祖宗。

左图：五次方程 x⁵ - x - 1 = 0，没有根式解。牛顿法从一个猜测出发，
      每一步用切线和 x 轴的交点当作新的猜测，红点一步步滑向真正的根。
右图：同一个"没有公式就迭代"的思想，换成在碗形 loss 上找最低点——
      这就是训练神经网络时 optimizer 干的事：沿坡往下挪一小步，再挪一小步。

运行: python no_formula_so_iterate_visualization.py
依赖: numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------
# 左：五次方程的牛顿迭代（没有求根公式，只能逼近）
# ---------------------------------------------------------------
def p(x):
    return x ** 5 - x - 1          # 著名的"解不出公式"的五次方程


def dp(x):
    return 5 * x ** 4 - 1


# 预先把牛顿法每一步算出来
newton_xs = [1.6]
for _ in range(7):
    x = newton_xs[-1]
    newton_xs.append(x - p(x) / dp(x))

# ---------------------------------------------------------------
# 右：梯度下降在碗形 loss 上找最低点（同一个迭代思想）
# ---------------------------------------------------------------
def f(w):
    return (w - 2.0) ** 2 + 1.0    # 碗底在 w=2，对应"最优权重"


def df(w):
    return 2 * (w - 2.0)


lr = 0.18
gd_ws = [-3.0]
for _ in range(7):
    w = gd_ws[-1]
    gd_ws.append(w - lr * df(w))   # optimizer.step() 的本体：沿负梯度挪一步

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
FRAMES = len(newton_xs)


def update(frame):
    # ---------- 左：牛顿法 ----------
    axL.clear()
    xx = np.linspace(-0.5, 2.0, 400)
    axL.axhline(0, color="#cccccc", lw=0.8)
    axL.plot(xx, p(xx), lw=2.2, color="#1f77b4", label="y = x⁵ - x - 1")

    x_cur = newton_xs[frame]
    axL.scatter([x_cur], [p(x_cur)], color="#d62728", zorder=5)
    axL.scatter([x_cur], [0], color="#d62728", marker="x", zorder=5)
    if frame + 1 < len(newton_xs):
        # 画当前点的切线，切线与 x 轴的交点就是下一个猜测
        x_next = newton_xs[frame + 1]
        tang_x = np.array([x_cur - 0.4, x_next + 0.05])
        tang_y = p(x_cur) + dp(x_cur) * (tang_x - x_cur)
        axL.plot(tang_x, tang_y, "--", color="#d62728", lw=1.4,
                 label="切线 → 交 x 轴得下一步")
    axL.set_xlim(-0.5, 2.0)
    axL.set_ylim(-3, 6)
    axL.set_title("左：五次方程没有求根公式，只能『沿切线逼近』\n"
                  f"第 {frame} 步  猜测 x={x_cur:.5f}  误差 p(x)={p(x_cur):+.5f}")
    axL.legend(fontsize=9, loc="upper left")

    # ---------- 右：梯度下降 ----------
    axR.clear()
    ww = np.linspace(-4, 4, 400)
    axR.plot(ww, f(ww), lw=2.2, color="#2ca02c", label="loss(w) = (w-2)²+1")
    w_cur = gd_ws[frame]
    axR.scatter([w_cur], [f(w_cur)], color="#d62728", zorder=5)
    # 走过的轨迹
    past = np.array(gd_ws[:frame + 1])
    axR.plot(past, f(past), "o-", color="#d62728", lw=1.2, ms=4, alpha=0.6)
    axR.scatter([2], [1], color="black", marker="*", s=120, zorder=4,
                label="碗底 = 最优权重")
    axR.set_xlim(-4, 4)
    axR.set_ylim(0, 26)
    axR.set_title("右：神经网络也没有『一步解出最优权重』的公式\n"
                  f"梯度下降第 {frame} 步  w={w_cur:.4f}（同一个迭代思想）")
    axR.legend(fontsize=9, loc="upper right")


anim = FuncAnimation(fig, update, frames=FRAMES, interval=900, repeat=True)

# 想存成 gif 取消下一行注释（需要 pillow）:
# anim.save("no_formula_so_iterate_visualization.gif", writer="pillow", fps=1)

plt.tight_layout()
plt.show()
