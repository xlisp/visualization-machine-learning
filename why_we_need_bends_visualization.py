"""
为什么需要"能弯的函数"：把弯一下当积木，拼出任意复杂曲线
========================================================

一句话要让你"看见"的事：
    一个 sigmoid / tanh，就是【一个弯】（一道台阶/一个拐弯）。
    单独一个弯啥也拟合不了；但把许多个【不同位置、不同高矮、不同朝向】的弯
    加权叠加起来，就能拼出任意复杂的曲线 —— 这就是"通用逼近定理"的画面，
    也是为什么神经网络非要在线性层之间插 sigmoid/tanh/ReLU。

  · 上图：训练好的小网络里，每个隐藏神经元贡献的【一个弯】(细线)
  · 下图：把这些弯全部加起来(蓝)，逼近一条又扭又绕的目标曲线(黑虚线)

运行: python why_we_need_bends_visualization.py
依赖: numpy matplotlib torch
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(0)

# ---------------------------------------------------------------
# 目标：一条又扭又绕的曲线（纯直线绝对拟合不了）
# ---------------------------------------------------------------
def target(x):
    return np.sin(2 * x) + 0.5 * np.sin(5 * x)


x_np = np.linspace(-3, 3, 400)
xt = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
yt = torch.tensor(target(x_np), dtype=torch.float32).unsqueeze(1)

# ---------------------------------------------------------------
# 一个只有 1 个隐藏层的小网络：Linear -> tanh -> Linear
#   隐藏层每个神经元 = 一个 tanh = 一个"弯"
#   输出层 = 把这些弯加权求和
# ---------------------------------------------------------------
H = 10
net = nn.Sequential(nn.Linear(1, H), nn.Tanh(), nn.Linear(H, 1))
opt = torch.optim.Adam(net.parameters(), lr=0.02)

for _ in range(4000):
    loss = ((net(xt) - yt) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"训练完成，loss = {loss.item():.4f}")

# 取出参数，手算每个隐藏神经元贡献的"那一个弯"
W1 = net[0].weight.detach().numpy()      # (H, 1)
b1 = net[0].bias.detach().numpy()        # (H,)
W2 = net[2].weight.detach().numpy()      # (1, H)
b2 = net[2].bias.detach().numpy()        # (1,)

# 第 j 个弯：bend_j(x) = W2[j] * tanh(W1[j]*x + b1[j])
bends = np.stack([W2[0, j] * np.tanh(W1[j, 0] * x_np + b1[j]) for j in range(H)])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

# ---------- 上：每个神经元就是"一个弯" ----------
for j in range(H):
    ax1.plot(x_np, bends[j], lw=1.4, alpha=0.8)
ax1.axhline(0, color="#cccccc", lw=0.8)
ax1.set_title(f"上：{H} 个隐藏神经元 = {H} 个『弯』（每个 tanh 一道拐弯，位置/高矮/朝向各不同）")

# ---------- 下：把所有弯加起来 ≈ 目标曲线 ----------
ax2.plot(x_np, target(x_np), "k--", lw=2.5, label="目标曲线（又扭又绕）")
ax2.plot(x_np, net(xt).detach().numpy().ravel(), color="#1f77b4", lw=2.2,
         label=f"{H} 个弯加权求和（网络输出）")
# 也画一条纯线性(一次)的最佳拟合，作为对照：直线弯不了
k, c = np.polyfit(x_np, target(x_np), 1)
ax2.plot(x_np, k * x_np + c, color="#d62728", lw=1.6, ls=":",
         label="纯线性(一次)最佳拟合 —— 一道弯都弯不出来")
ax2.axhline(0, color="#cccccc", lw=0.8)
ax2.set_title("下：弯 + 弯 + … 叠加 = 任意复杂曲线（这就是为什么必须有非线性）")
ax2.legend(fontsize=9, loc="upper right")

plt.tight_layout()

# 想存图取消下一行注释:
# plt.savefig("why_we_need_bends_visualization.png", dpi=130, bbox_inches="tight")

plt.show()
