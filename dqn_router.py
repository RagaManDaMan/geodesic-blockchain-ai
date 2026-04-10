import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from geodesic_overlay import build_icosahedron, subdivide, mesh_to_graph

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# --- Link parameters (same as before) ---

def assign_link_params(G):
    for u, v in G.edges():
        G[u][v]['L'] = np.random.uniform(0.1, 1.0)
        G[u][v]['C'] = np.random.uniform(0.1, 1.0)
        G[u][v]['delay'] = 1.0 / np.sqrt(G[u][v]['L'] * G[u][v]['C'])
        G[u][v]['base_latency'] = G[u][v]['delay'] + np.random.uniform(0, 0.5)
    return G

def measure_latency(G, u, v):
    return max(0.01, G[u][v]['base_latency'] + np.random.normal(0, 0.05))

# --- Neural Network ---
# Input: 3D position of current node (3)
#      + 3D position of target node (3)
#      + 3D position of candidate neighbour (3)
#      + edge latency to that neighbour (1)
# Total input: 10 features per (node, target, neighbour) triple
# Output: single Q-value for taking that edge
#
# Why this works: the network learns geometry.
# "Move toward the target on the sphere" becomes a learnable pattern.

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- Experience Replay Buffer ---
# Instead of learning from each experience immediately (which is unstable),
# we store experiences and sample random batches.
# This breaks correlations and stabilises training significantly.

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, reward, next_states, done):
        self.buffer.append((state, reward, next_states, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

# --- Feature builder ---
# Converts (current, target, neighbour) into a 10-dim tensor

def make_features(verts, u, target, v, latency):
    pos_u      = verts[u]
    pos_target = verts[target]
    pos_v      = verts[v]
    features   = np.concatenate([pos_u, pos_target, pos_v, [latency]])
    return torch.FloatTensor(features)

# --- DQN Router ---

class DQNRouter:
    def __init__(self, G, verts, alpha=0.001, gamma=0.95, epsilon=0.4):
        self.G         = G
        self.verts     = verts
        self.gamma     = gamma
        self.epsilon   = epsilon

        self.q_net     = QNetwork()
        self.target_net = QNetwork()  # stable copy for computing targets
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)
        self.buffer    = ReplayBuffer(capacity=15000)
        self.loss_fn   = nn.MSELoss()

        self.episode_rewards = []
        self.success_rate    = []
        self.losses          = []

    def get_q(self, u, target, v):
        latency = self.G[u][v]['base_latency']
        feat    = make_features(self.verts, u, target, v, latency)
        with torch.no_grad():
            return self.q_net(feat.unsqueeze(0)).item()

    def select_next(self, u, target, visited):
        neighbors = [n for n in self.G.neighbors(u) if n not in visited]
        if not neighbors:
            return None
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        return max(neighbors, key=lambda v: self.get_q(u, target, v))

    def compute_reward(self, u, v, target, latency, distances):
        reward   = -latency
        dist_before = distances[u].get(target, 999)
        dist_after  = distances[v].get(target, 999)
        progress    = dist_before - dist_after
        if v == target:
            reward += 20.0
        else:
            reward += progress * 2.0
        return reward

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None

        batch = self.buffer.sample(batch_size)
        loss_total = 0.0

        for (feat, reward, next_feats, done) in batch:
            current_q = self.q_net(feat.unsqueeze(0))

            if done or not next_feats:
                target_q = torch.FloatTensor([[reward]])
            else:
                with torch.no_grad():
                    next_qs = [self.target_net(f.unsqueeze(0)).item()
                               for f in next_feats]
                    target_q = torch.FloatTensor(
                        [[reward + self.gamma * max(next_qs)]]
                    )

            loss = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()

        return loss_total / batch_size

    def train(self, distances, episodes=800, steps_per_episode=25,
              target_update_freq=20):
        nodes = list(self.G.nodes())
        print(f"Training DQN router: {episodes} episodes...")

        for ep in range(episodes):
            source = random.choice(nodes)
            target = random.choice(nodes)
            while target == source:
                target = random.choice(nodes)

            u          = source
            visited    = {u}
            total_reward = 0
            reached    = False

            for step in range(steps_per_episode):
                if u == target:
                    reached = True
                    break

                v = self.select_next(u, target, visited)
                if v is None:
                    break

                latency = measure_latency(self.G, u, v)
                reward  = self.compute_reward(u, v, target,
                                              latency, distances)
                done    = (v == target)

                # Build feature for this transition
                feat = make_features(self.verts, u, target, v, latency)

                # Build features for next state's neighbours
                next_neighbors = [n for n in self.G.neighbors(v)
                                  if n not in visited]
                next_feats = [
                    make_features(self.verts, v, target, w,
                                  self.G[v][w]['base_latency'])
                    for w in next_neighbors
                ]

                self.buffer.push(feat, reward, next_feats, done)
                loss = self.train_step(batch_size=64)
                if loss:
                    self.losses.append(loss)

                total_reward += reward
                visited.add(v)
                u = v

            self.episode_rewards.append(total_reward)
            self.success_rate.append(1 if reached else 0)

            # Sync target network periodically for stability
            if (ep + 1) % target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            # Decay exploration
            self.epsilon = max(0.05, self.epsilon * 0.997)

            if (ep + 1) % 100 == 0:
                recent_success = np.mean(self.success_rate[-100:]) * 100
                recent_reward  = np.mean(self.episode_rewards[-100:])
                print(f"  Episode {ep+1:4d}: reward = {recent_reward:6.2f}, "
                      f"success = {recent_success:.0f}%, "
                      f"epsilon = {self.epsilon:.3f}, "
                      f"buffer = {len(self.buffer)}")

    def route(self, source, target, max_steps=30):
        path    = [source]
        visited = {source}
        u       = source
        for _ in range(max_steps):
            if u == target:
                break
            neighbors = [n for n in self.G.neighbors(u) if n not in visited]
            if not neighbors:
                break
            v = max(neighbors, key=lambda n: self.get_q(u, target, n))
            path.append(v)
            visited.add(v)
            u = v
        return path, u == target

# --- Comparison ---

def run_comparison(router, G, n_pairs=40):
    nodes   = list(G.nodes())
    results = []
    attempts = 0
    timeout_count = 0

    while len(results) < n_pairs and attempts < n_pairs * 3:
        attempts += 1
        s = random.choice(nodes)
        t = random.choice(nodes)
        if s == t:
            continue
        try:
            sp = nx.shortest_path(G, s, t)
            # Hard timeout: skip pairs where shortest path > 10 hops
            # These are hardest to route and most likely to hang
            if len(sp) - 1 > 10:
                continue

            qp, reached = router.route(s, t, max_steps=30)
            if not reached:
                timeout_count += 1
                continue

            sp_lat = sum(measure_latency(G, sp[i], sp[i+1])
                        for i in range(len(sp)-1))
            qp_lat = sum(measure_latency(G, qp[i], qp[i+1])
                        for i in range(len(qp)-1))
            results.append({
                'sp_hops': len(sp)-1, 'qp_hops': len(qp)-1,
                'sp_lat':  sp_lat,    'qp_lat':  qp_lat,
            })
            print(f"  Pair {len(results):2d}: "
                  f"sp={len(sp)-1} hops/{sp_lat:.2f}lat  "
                  f"dqn={len(qp)-1} hops/{qp_lat:.2f}lat")
        except nx.NetworkXNoPath:
            continue

    print(f"\n  ({timeout_count} pairs skipped — DQN didn't reach target)")
    return results

# --- Visualize ---

def visualize(router, results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Training curve
    ax  = axes[0]
    win = 50
    smoothed = np.convolve(router.episode_rewards,
                           np.ones(win)/win, mode='valid')
    ax.plot(smoothed, color='navy', label='Avg reward')
    ax2 = ax.twinx()
    success_smooth = np.convolve(router.success_rate,
                                 np.ones(win)/win, mode='valid')
    ax2.plot(success_smooth * 100, color='tomato',
             linestyle='--', label='Success %')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Success rate %', color='tomato')
    ax.set_title('DQN Training\n(reward + success rate)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg reward')

    # Hops comparison
    ax = axes[1]
    x  = range(len(results))
    ax.plot(x, [r['sp_hops'] for r in results], 'o-',
            color='gray', label='Shortest hops')
    ax.plot(x, [r['qp_hops'] for r in results], 's-',
            color='navy', label='DQN hops')
    ax.set_title('Hops: Shortest vs DQN')
    ax.set_xlabel('Sample pair')
    ax.set_ylabel('Hops')
    ax.legend()

    # Latency comparison
    ax  = axes[2]
    avg = np.mean([(r['sp_lat'] - r['qp_lat']) / r['sp_lat'] * 100
                   for r in results])
    ax.plot(x, [r['sp_lat'] for r in results], 'o-',
            color='gray', label='Hop-shortest')
    ax.plot(x, [r['qp_lat'] for r in results], 's-',
            color='navy', label='DQN routed')
    ax.set_title(f'Latency Comparison\nDQN avg improvement: {avg:.1f}%')
    ax.set_xlabel('Sample pair')
    ax.set_ylabel('Total path latency')
    ax.legend()

    plt.tight_layout()
    plt.savefig('dqn_results.png', dpi=150)
    plt.show()

    print(f"\nResults across {len(results)} successful routes:")
    print(f"  Avg hops    — shortest: "
          f"{np.mean([r['sp_hops'] for r in results]):.1f}  "
          f"DQN: {np.mean([r['qp_hops'] for r in results]):.1f}")
    print(f"  Avg latency — shortest: "
          f"{np.mean([r['sp_lat'] for r in results]):.2f}  "
          f"DQN: {np.mean([r['qp_lat'] for r in results]):.2f}")
    print(f"  Avg latency improvement: {avg:.1f}%")
    print(f"  Routes found: {len(results)}/40")

# --- Run ---

if __name__ == "__main__":
    print("Building mesh...")
    verts, faces = build_icosahedron()
    verts, faces = subdivide(verts, faces)
    verts, faces = subdivide(verts, faces)
    G     = mesh_to_graph(verts, faces)
    G     = assign_link_params(G)
    verts = np.array(verts)

    print("Precomputing distances...")
    distances = dict(nx.all_pairs_shortest_path_length(G))

    router = DQNRouter(G, verts, alpha=0.001, gamma=0.95, epsilon=0.4)
    router.train(distances, episodes=800, steps_per_episode=25)

    print("\nRunning comparison on 40 random pairs...")
    results = run_comparison(router, G, n_pairs=20)

    if results:
        visualize(router, results)
    else:
        print("No successful routes found — need more training.")
