"""
特征向量可视化（3Blue1Brown 风格）
====================================

一句话要让你"看见"的事：
    一个矩阵作用在空间上时，**绝大多数箭头都会被转离原来的方向**；
    只有少数几根特殊方向的箭头，**始终待在自己的那条线上，只被拉长/缩短** ——
    那几根方向就是【特征向量】，拉伸的倍数就是【特征值】。

画面里：
  · 灰色网格      —— 整个空间，随变换一起被揉斜（3b1b 的招牌画面）
  · 灰色细箭头    —— 一圈"探针"向量，看它们怎么被转离自己的虚线起始方向
  · 红 / 蓝粗箭头 —— 两个特征向量，它们死死贴在自己的虚线上，只变长，从不转向
  · 圆 -> 椭圆    —— 单位圆被变换成椭圆，椭圆的长短轴正好就是特征向量方向

运行: python eigenvector_visualization.py
依赖: numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------
# 1. 要演示的矩阵，以及它的特征值 / 特征向量
# ---------------------------------------------------------------
M = np.array([[2.0, 1.0],
              [1.0, 2.0]])
I = np.eye(2)

vals, vecs = np.linalg.eigh(M)        # 升序：特征值 [1, 3]
eig = [(vecs[:, i] / np.linalg.norm(vecs[:, i]), vals[i]) for i in range(2)]
EIG_COLORS = ["#1f77b4", "#d62728"]   # 蓝=特征值1，红=特征值3

# ---------------------------------------------------------------
# 2. 一圈"探针"箭头（普通方向），看它们被转离起始方向
#    （半圈就够，因为 v 和 -v 在同一条线上）
# ---------------------------------------------------------------
angles = np.linspace(0, np.pi, 12, endpoint=False)
probes = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 1.6

# 单位圆，用来看"圆 -> 椭圆"
th = np.linspace(0, 2 * np.pi, 120)
circle = np.stack([np.cos(th), np.sin(th)], axis=1)

# 背景网格线
g = np.linspace(-4, 4, 17)


def grid_lines(A):
    lines = []
    for c in g:
        vert = np.stack([np.full_like(g, c), g], axis=1) @ A.T   # 竖线 x=c
        hori = np.stack([g, np.full_like(g, c)], axis=1) @ A.T   # 横线 y=c
        lines += [vert, hori]
    return lines


def arrow(ax, vec, color, lw, alpha=1.0):
    ax.annotate("", xy=(vec[0], vec[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha))


# ---------------------------------------------------------------
# 3. 动画：t 从 0 平滑到 1 再回到 0（A(t) = I + t·(M − I)）
#    关键事实：M 的特征向量，对 A(t) 同样是特征向量，所以它们
#    在整个过程里【从不离开自己的那条线】—— 这正是要让你看见的。
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 7.5))
FRAMES = 120


def update(frame):
    t = 1 - abs(1 - 2 * frame / FRAMES)          # 0 -> 1 -> 0，方便循环观看
    A = I + t * (M - I)

    ax.clear()
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.axhline(0, color="#bbbbbb", lw=0.8); ax.axvline(0, color="#bbbbbb", lw=0.8)

    # 背景网格：整个空间被揉斜
    for line in grid_lines(A):
        ax.plot(line[:, 0], line[:, 1], color="#e0e0e0", lw=0.8, zorder=1)

    # 两条特征向量的"span"（那条永不改变的方向线，虚线）
    for (d, lam), col in zip(eig, EIG_COLORS):
        p = d * 7
        ax.plot([-p[0], p[0]], [-p[1], p[1]], "--", color=col, lw=1.0,
                alpha=0.5, zorder=2)

    # 单位圆 -> 椭圆
    ell = circle @ A.T
    ax.plot(ell[:, 0], ell[:, 1], color="#888888", lw=1.2, zorder=3)

    # 普通探针箭头：会被转离起始方向
    for p in probes:
        arrow(ax, p @ A.T, color="#999999", lw=1.6, alpha=0.9)

    # 特征向量箭头：始终贴在虚线上，只变长（长度因子 = 1 + t(λ−1)）
    for (d, lam), col in zip(eig, EIG_COLORS):
        scale = 1 + t * (lam - 1)
        arrow(ax, d * 1.6 * scale, color=col, lw=3.5)

    l1 = 1 + t * (eig[0][1] - 1)
    l3 = 1 + t * (eig[1][1] - 1)
    ax.set_title(
        "特征向量：普通箭头(灰)被转离自己的线；特征箭头(红/蓝)只拉伸不转向\n"
        f"变换进度 t={t:0.2f}   蓝向量长×{l1:0.2f}（特征值→1）   "
        f"红向量长×{l3:0.2f}（特征值→3）",
        fontsize=11)


anim = FuncAnimation(fig, update, frames=FRAMES, interval=60, repeat=True)

# 想存成 gif 取消下一行注释（需要 pillow）:
# anim.save("eigenvector_visualization.gif", writer="pillow", fps=20)

plt.show()
