"""
一张纸变形的故事：深度学习在学习"空间变换"
================================================

把一张脸画在一张橡皮纸上。捏一捏纸的不同部位，脸就有了不同表情 ——
微笑、沮丧、惊讶。每一种表情，都是对**整张纸的一次变形**（一次空间变换）。

这个脚本分两幕：

  第一幕（手动）：依次把同一张脸捏成 微笑 / 沮丧 / 惊讶，
                  让你先看清楚"表情 = 纸的变形 = 空间变换"。
  第二幕（学习）：随机给网络看一堆"点揉之前 / 揉之后的位置"，
                  让它把其中一种复合表情的变形规则**自己学出来**，
                  并且整张纸（网格线）跟着一起变形 —— 这就是
                  "矩阵就是映射、把整个空间搬动"的动态实证。

运行: python face_morph_paper_transform.py
依赖: torch numpy matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------
# 1. 表情 = 对整张纸的一次变形（一个作用在平面上的位移场）
#    捏不同的部位，就得到不同表情
# ---------------------------------------------------------------
def expr_field(p, kind, a):
    """把平面上的点 p 按 kind 表情、强度 a(0~1) 揉一下。"""
    x, y = p[:, 0].copy(), p[:, 1].copy()
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)

    mouth_mask = np.exp(-((y + 0.7) ** 2) / 0.18)        # 嘴巴附近的区域
    eye_mask = np.exp(-((y - 0.45) ** 2) / 0.10)         # 眼睛附近的区域

    if kind == "smile":                                  # 嘴角上扬
        dy += a * (0.9 * x ** 2 - 0.18) * mouth_mask
    elif kind == "sad":                                  # 嘴角下垂
        dy += a * (-(0.9 * x ** 2 - 0.18)) * mouth_mask
    elif kind == "surprise":                             # 张嘴 + 挑眉
        center = np.exp(-(x ** 2) / 0.30)
        dy += a * (-0.55) * mouth_mask * center          # 下巴往下掉
        dy += a * 0.22 * eye_mask                        # 眉眼上抬
    return np.stack([x + dx, y + dy], axis=1)


def target_warp(p):
    """第二幕让网络去学的"复合表情"：又惊又喜。"""
    q = expr_field(p, "smile", 1.0)
    q = expr_field(q, "surprise", 0.7)
    return q


# ---------------------------------------------------------------
# 2. 在纸上画一张脸（中性表情，嘴巴接近一条平线）
# ---------------------------------------------------------------
def make_face():
    t = np.linspace(0, 2 * np.pi, 200)
    head = np.stack([1.0 * np.cos(t), 1.3 * np.sin(t)], axis=1)
    eye_l = np.stack([-0.4 + 0.13 * np.cos(t), 0.45 + 0.13 * np.sin(t)], axis=1)
    eye_r = np.stack([0.4 + 0.13 * np.cos(t), 0.45 + 0.13 * np.sin(t)], axis=1)
    s = np.linspace(-0.55, 0.55, 80)
    mouth = np.stack([s, -0.7 + 0.05 * s ** 2], axis=1)   # 中性：接近平嘴
    return [head, eye_l, eye_r, mouth]


faceA = np.concatenate(make_face(), axis=0).astype(np.float32)   # 中性脸的所有点


# ---------------------------------------------------------------
# 3. 把"整张纸"画成网格线，看它怎么被一起揉变形
# ---------------------------------------------------------------
n = 15
lin = np.linspace(-1.7, 1.7, n)
gx, gy = np.meshgrid(lin, lin)
grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def grid_lines(points):
    """把 (n*n, 2) 的点还原成横竖网格线。"""
    pts = points.reshape(n, n, 2)
    return [pts[i] for i in range(n)] + [pts[:, j] for j in range(n)]


def draw(ax, warped_grid, warped_face, title, ref=None):
    ax.clear()
    if ref is not None:                                   # 目标脸（淡虚线参考）
        ax.scatter(ref[:, 0], ref[:, 1], s=6, c="#dddddd")
    for line in grid_lines(warped_grid):
        ax.plot(line[:, 0], line[:, 1], c="#9ecae1", lw=0.8)
    ax.scatter(warped_face[:, 0], warped_face[:, 1], s=9, c="#d62728")
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.8, 2.8)
    ax.set_aspect("equal")


# ---------------------------------------------------------------
# 4. 第二幕用的网络 + 数据（学 target_warp 这套揉纸规则）
# ---------------------------------------------------------------
X = np.random.uniform(-1.7, 1.7, size=(3000, 2)).astype(np.float32)
Y = target_warp(X).astype(np.float32)
Xt, Yt = torch.tensor(X), torch.tensor(Y)
grid_t, faceA_t = torch.tensor(grid), torch.tensor(faceA)
faceB = target_warp(faceA).astype(np.float32)

net = nn.Sequential(
    nn.Linear(2, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 2),
)
opt = torch.optim.Adam(net.parameters(), lr=0.01)


# ---------------------------------------------------------------
# 5. 动画：第一幕表情切换 -> 第二幕训练
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6.5))

INTRO = [("smile", "第一幕 ①  微笑 = 捏一下嘴角"),
         ("sad", "第一幕 ②  沮丧 = 反向捏嘴角"),
         ("surprise", "第一幕 ③  惊讶 = 张嘴 + 挑眉")]
PER = 28                       # 每个表情：上扬12 + 保持4 + 收回12
INTRO_FRAMES = PER * len(INTRO)
LEARN_FRAMES = 90
STEPS_PER_FRAME = 40


def amount(local):
    if local < 12:
        return local / 12
    if local < 16:
        return 1.0
    return max(0.0, 1 - (local - 16) / 12)


def update(frame):
    if frame < INTRO_FRAMES:                              # ---- 第一幕 ----
        kind, label = INTRO[frame // PER]
        a = amount(frame % PER)
        draw(ax, expr_field(grid, kind, a),
             expr_field(faceA, kind, a),
             f"{label}\n（表情 = 整张纸的一次变形）")
    else:                                                 # ---- 第二幕 ----
        for _ in range(STEPS_PER_FRAME):
            loss = ((net(Xt) - Yt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        step = (frame - INTRO_FRAMES) * STEPS_PER_FRAME
        with torch.no_grad():
            draw(ax, net(grid_t).numpy(), net(faceA_t).numpy(),
                 f"第二幕  深度学习在自学这套变换\nstep={step}  loss={loss.item():.4f}",
                 ref=faceB)


anim = FuncAnimation(fig, update, frames=INTRO_FRAMES + LEARN_FRAMES,
                     interval=80, repeat=False)

# 想存成 gif 取消下一行注释（需要 pillow）:
# anim.save("face_morph_paper_transform.gif", writer="pillow", fps=12)

plt.show()
