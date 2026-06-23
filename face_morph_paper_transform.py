"""
一张纸变形的故事：深度学习在学习"空间变换"
================================================

把一张人脸画在橡皮纸上。现在有人按某种固定的方式把这张纸揉皱、拉扯，
脸也就跟着变成了另一张脸。这个"怎么揉"的规则，就是一个空间变换。

深度学习要干的事：只给它"纸上每个点揉之前 / 揉之后的位置"，
让一个神经网络把这套揉纸规则**自己学出来**。学会以后，
不仅脸被正确地变成目标脸，连整张纸（网格线）都跟着一起变形 ——
这就是"矩阵就是映射、把整个空间搬动"这句话的动态版本。

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
# 1. "真正的揉纸规则"：把平面上任意一点搬到新位置（人为设定，网络看不到）
#    这是一个非线性变换 —— 真实人脸之间的变化也不是简单的旋转拉伸
# ---------------------------------------------------------------
def true_warp(p):
    x, y = p[:, 0], p[:, 1]
    nx = x + 0.35 * np.sin(1.5 * y) + 0.10 * y
    ny = y + 0.30 * np.sin(1.3 * x) - 0.12 * x
    return np.stack([nx, ny], axis=1)


# ---------------------------------------------------------------
# 2. 在纸上画一张笑脸（脸只是画在纸上的图案，会被揉纸规则一起带走）
# ---------------------------------------------------------------
def make_face():
    t = np.linspace(0, 2 * np.pi, 200)
    head = np.stack([1.0 * np.cos(t), 1.3 * np.sin(t)], axis=1)
    eye_l = np.stack([-0.4 + 0.13 * np.cos(t), 0.45 + 0.13 * np.sin(t)], axis=1)
    eye_r = np.stack([0.4 + 0.13 * np.cos(t), 0.45 + 0.13 * np.sin(t)], axis=1)
    s = np.linspace(-0.55, 0.55, 80)
    mouth = np.stack([s, -0.45 + 0.5 * s ** 2 - 0.25], axis=1)   # 上扬的微笑
    return [head, eye_l, eye_r, mouth]


face_parts = make_face()
faceA = np.concatenate(face_parts, axis=0).astype(np.float32)   # 原始脸的所有点
faceB = true_warp(faceA).astype(np.float32)                     # 揉过之后的目标脸


# ---------------------------------------------------------------
# 3. 训练数据：纸上随机撒一堆点，记录它们"揉之前 -> 揉之后"的位置
#    这就是深度学习能看到的全部信息
# ---------------------------------------------------------------
X = np.random.uniform(-1.7, 1.7, size=(3000, 2)).astype(np.float32)
Y = true_warp(X).astype(np.float32)
Xt, Yt = torch.tensor(X), torch.tensor(Y)


# ---------------------------------------------------------------
# 4. 一个小网络，去学这套揉纸规则（输入 2 维坐标 -> 输出 2 维新坐标）
# ---------------------------------------------------------------
net = nn.Sequential(
    nn.Linear(2, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 2),
)
opt = torch.optim.Adam(net.parameters(), lr=0.01)


# ---------------------------------------------------------------
# 5. 把"整张纸"画成网格线，看它怎么被一起揉变形
# ---------------------------------------------------------------
n = 15
lin = np.linspace(-1.7, 1.7, n)
gx, gy = np.meshgrid(lin, lin)
grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
grid_t = torch.tensor(grid)
faceA_t = torch.tensor(faceA)


def warp_grid_lines(points):
    """把 (n*n, 2) 的点还原成横竖网格线，便于画出纸的变形。"""
    pts = points.reshape(n, n, 2)
    rows = [pts[i] for i in range(n)]      # 横线
    cols = [pts[:, j] for j in range(n)]   # 竖线
    return rows + cols


# ---------------------------------------------------------------
# 6. 动画：每帧训练若干步，画出当前的纸变形 + 脸变形
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6.5))
STEPS_PER_FRAME = 40
FRAMES = 80


def update(frame):
    for _ in range(STEPS_PER_FRAME):
        pred = net(Xt)
        loss = ((pred - Yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    ax.clear()
    with torch.no_grad():
        warped_grid = net(grid_t).numpy()
        warped_face = net(faceA_t).numpy()

    # 目标脸（淡虚线参考）
    ax.scatter(faceB[:, 0], faceB[:, 1], s=6, c="#cccccc", label="目标脸 B (参考)")
    # 当前被网络揉出来的纸
    for line in warp_grid_lines(warped_grid):
        ax.plot(line[:, 0], line[:, 1], c="#9ecae1", lw=0.8)
    # 当前被网络揉出来的脸
    ax.scatter(warped_face[:, 0], warped_face[:, 1], s=8, c="#d62728",
               label="网络当前揉出的脸")

    ax.set_title(f"深度学习在学'揉纸规则'  step={frame * STEPS_PER_FRAME}  "
                 f"loss={loss.item():.4f}")
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.8, 2.8)
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)


anim = FuncAnimation(fig, update, frames=FRAMES, interval=80, repeat=False)

# 想存成 gif 取消下一行注释（需要 pillow）:
# anim.save("face_morph_paper_transform.gif", writer="pillow", fps=12)

plt.show()
