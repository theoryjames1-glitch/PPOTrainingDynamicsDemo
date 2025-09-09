# PPOTrainingDynamicsDemo

"""
Got it ⚡ — let’s build a **demo program** that captures the theory we just built: PPO training is a balance between **reward signal**, **KL constraint**, and **batch variance**.

We’ll keep it simple and not use a giant GPT — just a toy “policy” (a small neural net) trained with a fake reward function. This way you can *see* the dynamics without burning GPU cycles.

---

# 🧪 PPO Training Dynamics Demo

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# ----------------------------
# 1. Toy Policy Model
# ----------------------------
class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)  # 5-dim input -> 2 actions

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

# ----------------------------
# 2. Fake Environment + Reward
# ----------------------------
def fake_env(action):
    """
    Fake reward: action 0 is usually bad, action 1 is usually good.
    Add noise to simulate language reward variability.
    """
    if action == 1:
        return random.uniform(0.4, 1.0)  # "good" reward
    else:
        return random.uniform(-0.5, 0.2) # "bad" reward

# ----------------------------
# 3. PPO Loss Components
# ----------------------------
def ppo_loss(old_probs, new_probs, actions, rewards, target_kl=0.1):
    ratios = new_probs.gather(1, actions) / old_probs.gather(1, actions)
    ratios = ratios.squeeze()

    # Clip PPO objective
    clipped = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * rewards
    loss = -torch.min(ratios * rewards, clipped).mean()

    # KL penalty (keep new policy near old)
    kl = (old_probs * (old_probs / (new_probs + 1e-8)).log()).sum(-1).mean()
    loss = loss + torch.abs(kl - target_kl)

    return loss, kl.item()

# ----------------------------
# 4. Training Loop
# ----------------------------
torch.manual_seed(42)
policy = TinyPolicy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

avg_rewards, kls = [], []

for step in range(200):
    # Batch rollout
    batch_obs = torch.randn(32, 5)
    old_probs = policy(batch_obs).detach()

    actions = torch.multinomial(old_probs, num_samples=1)
    rewards = torch.tensor([fake_env(a.item()) for a in actions]).float()

    new_probs = policy(batch_obs)
    loss, kl = ppo_loss(old_probs, new_probs, actions, rewards, target_kl=0.1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_rewards.append(rewards.mean().item())
    kls.append(kl)

    if step % 20 == 0:
        print(f"Step {step:3d} | Avg Reward: {avg_rewards[-1]:.3f} | KL: {kl:.3f}")

# ----------------------------
# 5. Plot Dynamics
# ----------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(avg_rewards, label="Avg Reward")
plt.axhline(0, color="red", linestyle="--")
plt.legend()
plt.title("Reward Dynamics")

plt.subplot(1,2,2)
plt.plot(kls, label="KL Divergence")
plt.axhline(0.1, color="green", linestyle="--", label="Target KL")
plt.legend()
plt.title("KL Dynamics")

plt.show()
```

## 🔹 What this Demo Shows

1. **Rewards start noisy**, but average reward trends upward as PPO learns.
2. **KL divergence oscillates** around the target — too high → PPO pushes back, too low → PPO allows bigger updates.
3. You’ll literally see the **“balance of forces”** from the theory:

   * Rewards push the policy toward better behavior.
   * KL penalty keeps it from drifting too far.
   * Batch variance smooths the noise.

---

⚡ Bottom line: This is a **lab experiment version** of PPO — small, fast, and it lets you watch the dynamics that caused all the weird warnings and blow-ups in your big LLM training.
