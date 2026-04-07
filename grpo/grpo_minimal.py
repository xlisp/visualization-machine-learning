"""
Minimal GRPO (Group Relative Policy Optimization) example with step-by-step
visualization.

This mirrors the pattern used in MathGPT (../MathGPT/scripts/train_rl.py) on a
tiny contextual-bandit "math" task so that every stage of GRPO can be plotted.

MathGPT's GRPO recipe (see train_rl.py):
    1. For each prompt, sample N rollouts from the policy
       (train_rl.py lines 126-137, engine.generate_batch)
    2. Score each rollout with a reward function
       (train_rl.py lines 140-143, task.reward -> 1.0 / 0.0)
    3. Advantage = reward - group mean (group-relative baseline)
       (train_rl.py line 158:  advantages = rewards - rewards.mean())
    4. Policy-gradient loss:  - E[ log pi(a|s) * A ]
       (train_rl.py lines 254-258: logp * advantages -> pg_obj -> -pg_obj)
    5. Gradient step; repeat.

Our toy task: prompt p in {0,1,2,3} -> correct token target(p) = (3*p + 2) % V.
Reward = 1 if sampled token matches target, else 0. Policy = per-prompt logits.

Run:
    python grpo_minimal.py

Outputs (all saved next to this file):
    grpo_01_task.png              - task definition
    grpo_02_policy_init.png       - policy before training
    grpo_03_rollouts.png          - group rollouts & rewards at step 0
    grpo_04_advantages.png        - reward vs group-relative advantage
    grpo_05_training_curves.png   - reward / loss / entropy over training
    grpo_06_policy_final.png      - learned policy
    grpo_06b_policy_evolution.png - snapshots of policy through training
    grpo_overview.png             - everything combined
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------------------------
# 1. Task: toy contextual bandit that plays the role of GSM8K in MathGPT.
# -----------------------------------------------------------------------------
NUM_PROMPTS = 4      # number of distinct "math problems"
VOCAB       = 8      # number of actions / tokens
TARGET = torch.tensor([(3 * p + 2) % VOCAB for p in range(NUM_PROMPTS)])

def reward_fn(prompt_idx: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """1.0 if the sampled action matches the prompt's target token, else 0.0.
    This is the analogue of tasks.gsm8k.GSM8K.reward in the MathGPT project."""
    return (action == TARGET[prompt_idx]).float()


# -----------------------------------------------------------------------------
# 2. Policy: smallest parametric policy that still has a gradient.
#    A table of logits[prompt, action]. Analogue of the LLM in MathGPT.
# -----------------------------------------------------------------------------
class TinyPolicy(nn.Module):
    def __init__(self, num_prompts: int, vocab: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_prompts, vocab))

    def forward(self, prompt_idx: torch.Tensor) -> torch.Tensor:
        return self.logits[prompt_idx]

    def probs(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=-1)


policy = TinyPolicy(NUM_PROMPTS, VOCAB)


# -----------------------------------------------------------------------------
# GRPO training step — directly mirrors MathGPT train_rl.py
# -----------------------------------------------------------------------------
NUM_SAMPLES = 8      # rollouts per prompt (the "group") — like --num-samples
LR          = 0.3
NUM_STEPS   = 120

optimizer = torch.optim.SGD(policy.parameters(), lr=LR)


def grpo_step():
    """One GRPO optimization step across all prompts.

    This is the toy version of the per-step logic in MathGPT train_rl.py.
    """
    # 1. Build a batch: every prompt contributes NUM_SAMPLES rollouts.
    prompts = torch.arange(NUM_PROMPTS).repeat_interleave(NUM_SAMPLES)   # [P*N]

    # 2. Sample actions from pi(.|prompt)  -- analogue of engine.generate_batch.
    logits  = policy(prompts)                                            # [P*N, V]
    dist    = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()                                              # [P*N]
    logp    = dist.log_prob(actions)                                     # [P*N]

    # 3. Score rollouts — analogue of task.reward().
    rewards = reward_fn(prompts, actions)                                # [P*N]

    # 4. Group-relative advantage: subtract per-prompt group mean.
    #    MathGPT uses a single global group; here we use one group per prompt
    #    (per-prompt baseline) which reduces variance further.
    rewards_by_group = rewards.view(NUM_PROMPTS, NUM_SAMPLES)
    baseline         = rewards_by_group.mean(dim=1, keepdim=True)        # [P, 1]
    advantages       = (rewards_by_group - baseline).view(-1)            # [P*N]

    # 5. Policy gradient loss:  maximize  logp * A   <=>   minimize  -(logp*A).
    loss = -(logp * advantages.detach()).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        entropy = dist.entropy().mean().item()
    return {
        "prompts":     prompts,
        "actions":     actions,
        "rewards":     rewards,
        "advantages":  advantages,
        "logp":        logp,
        "loss":        loss.item(),
        "mean_reward": rewards.mean().item(),
        "entropy":     entropy,
    }


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def draw_policy(ax, probs, title):
    im = ax.imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for p in range(NUM_PROMPTS):
        ax.text(TARGET[p].item(), p, "*", ha="center", va="center",
                color="crimson", fontsize=16, fontweight="bold")
    ax.set_xticks(range(VOCAB))
    ax.set_yticks(range(NUM_PROMPTS))
    ax.set_xlabel("token id (action)")
    ax.set_ylabel("prompt id")
    ax.set_title(title)
    return im


# -----------------------------------------------------------------------------
# STEP 1 — the task
# -----------------------------------------------------------------------------
print("[step 1] task")
task_grid = np.zeros((NUM_PROMPTS, VOCAB))
for p in range(NUM_PROMPTS):
    task_grid[p, TARGET[p]] = 1.0

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.imshow(task_grid, cmap="Greens", aspect="auto")
for p in range(NUM_PROMPTS):
    ax.text(TARGET[p].item(), p, "*", ha="center", va="center",
            color="white", fontsize=16, fontweight="bold")
ax.set_xticks(range(VOCAB))
ax.set_yticks(range(NUM_PROMPTS))
ax.set_xlabel("token id (action)")
ax.set_ylabel("prompt id")
ax.set_title("Step 1. Task - reward=1 only on the starred cell")
save(fig, "grpo_01_task.png")


# -----------------------------------------------------------------------------
# STEP 2 — initial policy (uniform because logits start at zero)
# -----------------------------------------------------------------------------
print("[step 2] initial policy")
with torch.no_grad():
    init_probs = policy.probs().clone().numpy()

fig, ax = plt.subplots(figsize=(6, 3.2))
im = draw_policy(ax, init_probs, "Step 2. Initial policy pi(a|prompt) - uniform")
fig.colorbar(im, ax=ax, label="pi(a|prompt)")
save(fig, "grpo_02_policy_init.png")


# -----------------------------------------------------------------------------
# STEP 3 & 4 — run the first GRPO step and visualize its internals
# -----------------------------------------------------------------------------
print("[step 3-4] first training step")
history = {"reward": [], "loss": [], "entropy": []}
snapshots = [("init", init_probs)]

first = grpo_step()
history["reward"].append(first["mean_reward"])
history["loss"].append(first["loss"])
history["entropy"].append(first["entropy"])

# -- Visualization 3: group rollouts + rewards --------------------------------
fig, axes = plt.subplots(NUM_PROMPTS, 1, figsize=(6.5, 6), sharex=True)
for p in range(NUM_PROMPTS):
    acts = first["actions"][p*NUM_SAMPLES:(p+1)*NUM_SAMPLES].numpy()
    rews = first["rewards"][p*NUM_SAMPLES:(p+1)*NUM_SAMPLES].numpy()
    colors = ["#2ca02c" if r > 0 else "#d62728" for r in rews]
    axes[p].bar(range(NUM_SAMPLES), acts, color=colors)
    axes[p].axhline(TARGET[p].item(), ls="--", color="gold", lw=1.5,
                    label=f"target = {TARGET[p].item()}")
    axes[p].set_ylim(-0.5, VOCAB - 0.5)
    axes[p].set_ylabel(f"prompt {p}\naction")
    axes[p].legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("rollout index within the group")
fig.suptitle(f"Step 3. Group rollouts (green = reward 1, red = reward 0)\n"
             f"mean reward at step 0 = {first['mean_reward']:.3f}")
save(fig, "grpo_03_rollouts.png")

# -- Visualization 4: reward vs group-relative advantage ----------------------
idx  = np.arange(NUM_PROMPTS * NUM_SAMPLES)
rews = first["rewards"].numpy()
advs = first["advantages"].numpy()

fig, ax = plt.subplots(figsize=(8, 3.6))
w = 0.4
ax.bar(idx - w/2, rews, w, label="reward", color="#1f77b4")
ax.bar(idx + w/2, advs, w, label="advantage = r - mean(r | group)", color="#ff7f0e")
ax.axhline(0, color="black", lw=0.6)
for p in range(1, NUM_PROMPTS):
    ax.axvline(p*NUM_SAMPLES - 0.5, color="gray", ls=":", lw=1)
for p in range(NUM_PROMPTS):
    ax.text(p*NUM_SAMPLES + NUM_SAMPLES/2 - 0.5, 1.08, f"prompt {p}",
            ha="center", fontsize=9, color="gray")
ax.set_ylim(-1.1, 1.25)
ax.set_xlabel("rollout (grouped by prompt)")
ax.set_title("Step 4. Group-relative advantage centers each group at 0")
ax.legend(loc="lower right")
save(fig, "grpo_04_advantages.png")


# -----------------------------------------------------------------------------
# Run the remaining training steps
# -----------------------------------------------------------------------------
print("[step 5] training loop")
SNAPSHOT_STEPS = {5, 20, 60, NUM_STEPS - 1}
for step in range(1, NUM_STEPS):
    info = grpo_step()
    history["reward"].append(info["mean_reward"])
    history["loss"].append(info["loss"])
    history["entropy"].append(info["entropy"])
    if step in SNAPSHOT_STEPS:
        with torch.no_grad():
            snapshots.append((step, policy.probs().clone().numpy()))


# -----------------------------------------------------------------------------
# STEP 5 — training curves
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
axes[0].plot(history["reward"], color="#2ca02c")
axes[0].set_title("mean reward")
axes[0].set_xlabel("step")
axes[0].set_ylim(-0.05, 1.05)
axes[0].grid(alpha=0.3)

axes[1].plot(history["loss"], color="#d62728")
axes[1].set_title("policy-gradient loss")
axes[1].set_xlabel("step")
axes[1].grid(alpha=0.3)

axes[2].plot(history["entropy"], color="#9467bd")
axes[2].set_title("policy entropy")
axes[2].set_xlabel("step")
axes[2].grid(alpha=0.3)

fig.suptitle("Step 5. Training curves")
fig.tight_layout()
save(fig, "grpo_05_training_curves.png")


# -----------------------------------------------------------------------------
# STEP 6 — final policy + evolution snapshots
# -----------------------------------------------------------------------------
print("[step 6] learned policy")
with torch.no_grad():
    final_probs = policy.probs().clone().numpy()

fig, ax = plt.subplots(figsize=(6, 3.2))
im = draw_policy(ax, final_probs, "Step 6. Learned policy - mass on the stars")
fig.colorbar(im, ax=ax, label="pi(a|prompt)")
save(fig, "grpo_06_policy_final.png")

# policy evolution grid
fig, axes = plt.subplots(1, len(snapshots), figsize=(2.7 * len(snapshots), 3.2))
for ax, (tag, probs) in zip(axes, snapshots):
    ax.imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for p in range(NUM_PROMPTS):
        ax.text(TARGET[p].item(), p, "*", ha="center", va="center",
                color="crimson", fontsize=12, fontweight="bold")
    ax.set_title(f"step {tag}")
    ax.set_xticks(range(VOCAB))
    ax.set_yticks(range(NUM_PROMPTS))
fig.suptitle("Policy evolution: probability mass migrates onto the correct token")
save(fig, "grpo_06b_policy_evolution.png")


# -----------------------------------------------------------------------------
# Everything in one figure
# -----------------------------------------------------------------------------
print("[overview]")
fig = plt.figure(figsize=(14, 9.5))
gs = fig.add_gridspec(3, 3, hspace=0.65, wspace=0.45)

# (1) task
ax = fig.add_subplot(gs[0, 0])
ax.imshow(task_grid, cmap="Greens", aspect="auto")
for p in range(NUM_PROMPTS):
    ax.text(TARGET[p].item(), p, "*", ha="center", va="center",
            color="white", fontsize=13, fontweight="bold")
ax.set_title("1. Task (reward=1 on *)")
ax.set_xlabel("action"); ax.set_ylabel("prompt")

# (2) initial policy
ax = fig.add_subplot(gs[0, 1])
draw_policy(ax, init_probs, "2. Initial pi(a|prompt)")

# (3) final policy
ax = fig.add_subplot(gs[0, 2])
draw_policy(ax, final_probs, "6. Learned pi(a|prompt)")

# (4) rollouts at step 0
ax = fig.add_subplot(gs[1, 0])
colors = ["#2ca02c" if r > 0 else "#d62728" for r in first["rewards"].numpy()]
ax.bar(range(len(first["actions"])), first["actions"].numpy(), color=colors)
for p in range(1, NUM_PROMPTS):
    ax.axvline(p*NUM_SAMPLES - 0.5, color="gray", ls=":")
ax.set_title("3. Rollouts at step 0")
ax.set_xlabel("rollout"); ax.set_ylabel("sampled action")

# (5) advantages
ax = fig.add_subplot(gs[1, 1])
ax.bar(idx - 0.2, rews, 0.4, label="reward", color="#1f77b4")
ax.bar(idx + 0.2, advs, 0.4, label="advantage", color="#ff7f0e")
for p in range(1, NUM_PROMPTS):
    ax.axvline(p*NUM_SAMPLES - 0.5, color="gray", ls=":")
ax.axhline(0, color="black", lw=0.6)
ax.set_title("4. r  vs  r - mean(r|group)")
ax.set_xlabel("rollout")
ax.legend(fontsize=7, loc="lower right")

# (6) training curves
ax = fig.add_subplot(gs[1, 2])
ax.plot(history["reward"],  label="reward",  color="#2ca02c")
ax.plot(history["entropy"], label="entropy", color="#9467bd")
ax.set_title("5. Training curves")
ax.set_xlabel("step")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# bottom row: policy evolution
bottom_snaps = [snapshots[0], snapshots[len(snapshots)//2], snapshots[-1]]
for i, (tag, probs) in enumerate(bottom_snaps):
    ax = fig.add_subplot(gs[2, i])
    ax.imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for p in range(NUM_PROMPTS):
        ax.text(TARGET[p].item(), p, "*", ha="center", va="center",
                color="crimson", fontsize=11, fontweight="bold")
    ax.set_title(f"pi at step {tag}")
    ax.set_xlabel("action"); ax.set_ylabel("prompt")

fig.suptitle(
    "Minimal GRPO:  prompt -> group rollouts -> reward -> "
    "group-relative advantage -> pg update",
    fontsize=13,
)
save(fig, "grpo_overview.png")

print(f"\nfinal mean reward (last 10 steps): "
      f"{np.mean(history['reward'][-10:]):.3f}")
print("done.")
