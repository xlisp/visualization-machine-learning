"""
从方程到矩阵：一张图看懂"次数越高，曲线越能弯"
================================================

一句话要让你"看见"的事：
    方程的【次数】，决定了曲线能弯到什么程度，也决定了"它好不好解"。

  · 一次 y = ax + b        —— 一条直线，一步就能解出根（x = -b/a）
  · 二次 y = ax² + bx + c  —— 一条抛物线（碗），配方法把它写成 a(x-h)²+k，
                              碗底 (h,k) 就是最小值 —— 这正是 loss 的形状
  · 三次 y = x³ ...        —— 出现"拐点(inflection)"，曲线开始呈"躺着的 S"，
                              这正是 sigmoid / tanh 这些激活函数的形状直觉
  · 二次型 z = xᵀAx        —— 把抛物线升到高维，就是深度学习里那个 loss 碗

四个子图分别对应文章里的四站，跑一遍就能把"代数→线性代数→激活函数"串起来。

运行: python from_equation_to_matrix_visualization.py
依赖: numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (注册 3d 投影)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


fig = plt.figure(figsize=(13, 10))

# ---------------------------------------------------------------
# 子图 1：一次 / 二次 / 三次 —— 次数越高，曲线越能弯
# ---------------------------------------------------------------
ax1 = fig.add_subplot(2, 2, 1)
x = np.linspace(-3, 3, 400)
ax1.axhline(0, color="#cccccc", lw=0.8)
ax1.axvline(0, color="#cccccc", lw=0.8)
ax1.plot(x, 0.8 * x + 0.5, label="一次 y=ax+b（直线）", lw=2.2)
ax1.plot(x, x ** 2 - 2, label="二次 y=x²-2（抛物线/碗）", lw=2.2)
ax1.plot(x, x ** 3 - x, label="三次 y=x³-x（躺着的 S）", lw=2.2)
ax1.set_ylim(-6, 6)
ax1.set_title("① 方程的『次数』= 曲线能弯几道\n一次=0 个弯，二次=1 个弯，三次=2 个弯（含拐点）")
ax1.legend(fontsize=9, loc="upper left")

# ---------------------------------------------------------------
# 子图 2：配方法 —— 把抛物线写成顶点式，碗底就是最小值（loss 的祖宗）
#     y = x² - 4x + 5 = (x-2)² + 1，顶点 (2,1)
# ---------------------------------------------------------------
ax2 = fig.add_subplot(2, 2, 2)
y = x ** 2 - 4 * x + 5
ax2.plot(x + 2, y, lw=2.4, color="#d62728")  # 平移让画面居中
h, k = 2, 1
ax2.scatter([h], [k], color="black", zorder=5)
ax2.annotate("碗底 = 最小值 (2,1)\n配方法: x²-4x+5 = (x-2)²+1",
             xy=(h, k), xytext=(h + 0.3, k + 6),
             arrowprops=dict(arrowstyle="-|>", color="black"), fontsize=9)
ax2.set_xlim(-1, 5)
ax2.set_ylim(0, 14)
ax2.set_title("② 求根公式背后的『配方法』\n= 找碗底 = 求 loss 最小值")

# ---------------------------------------------------------------
# 子图 3：三次的"躺 S" vs sigmoid / tanh —— 激活函数的形状直觉
# ---------------------------------------------------------------
ax3 = fig.add_subplot(2, 2, 3)
xs = np.linspace(-4, 4, 400)
ax3.axhline(0, color="#cccccc", lw=0.8)
ax3.axvline(0, color="#cccccc", lw=0.8)
ax3.plot(xs, np.tanh(xs), label="tanh（温柔的躺 S）", lw=2.4)
ax3.plot(xs, 1 / (1 + np.exp(-xs)), label="sigmoid（躺 S）", lw=2.4)
# 把三次曲线压缩一下，凸显它和 S 形同样有"拐点"
ax3.plot(xs, np.clip(0.06 * xs ** 3, -1.2, 1.2),
         "--", label="三次 0.06·x³（也有拐点）", lw=1.8, color="#7f7f7f")
ax3.set_ylim(-1.4, 1.4)
ax3.set_title("③ 三次曲线的『拐点』→ 激活函数的 S 形\nsigmoid/tanh 就是被驯服过的高次曲线")
ax3.legend(fontsize=9, loc="upper left")

# ---------------------------------------------------------------
# 子图 4：二次型 z = xᵀAx —— 把抛物线升到高维，就是 loss 碗
# ---------------------------------------------------------------
ax4 = fig.add_subplot(2, 2, 4, projection="3d")
A = np.array([[2.0, 0.5],
              [0.5, 1.0]])  # 正定 → 向上的碗
gx = np.linspace(-2, 2, 60)
gy = np.linspace(-2, 2, 60)
GX, GY = np.meshgrid(gx, gy)
# z = [x,y] A [x,y]^T = A00 x² + (A01+A10) xy + A11 y²
Z = A[0, 0] * GX ** 2 + (A[0, 1] + A[1, 0]) * GX * GY + A[1, 1] * GY ** 2
ax4.plot_surface(GX, GY, Z, cmap="viridis", alpha=0.9, linewidth=0)
ax4.scatter([0], [0], [0], color="red", s=40)
ax4.set_title("④ 二次型 z = x^T·A·x（向量版的抛物线）\n= 深度学习里的 loss 碗，碗底就是最优权重")

plt.tight_layout()

# 想存图取消下一行注释:
# plt.savefig("from_equation_to_matrix_visualization.png", dpi=130, bbox_inches="tight")

plt.show()
