import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from geodesic_overlay import build_icosahedron, subdivide, mesh_to_graph

random.seed(42)
np.random.seed(42)

def assign_link_params(G):
    for u, v in G.edges():
        G[u][v]['L'] = np.random.uniform(0.1, 1.0)
        G[u][v]['C'] = np.random.uniform(0.1, 1.0)
        G[u][v]['delay'] = 1.0 / np.sqrt(G[u][v]['L'] * G[u][v]['C'])
        G[u][v]['base_latency'] = G[u][v]['delay'] + np.random.uniform(0, 0.5)
    return G

def measure_latency(G, u, v):
    base = G[u][v]['base_latency']
    jitter = np.random.normal(0, 0.05)
    return max(0.01, base + jitter)

# --- Precompute shortest path lengths for distance-to-target reward shaping ---
# This is what the patent is missing. Without knowing how far you are
# from the target, the agent can't learn to navigate toward it.

def precompute_distances(G):
    print("Precomputing all-pairs shortest path lengths...")
    return dict(nx.all_pairs_shortest_path_length(G))

class GoalConditionedRouter:
    def __init__(self, G, distances, alpha=0.15, gamma=0.95, epsilon=0.3):
        self.G = G
        self.distances = distances
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # State is now (current, target) pair — destination-aware
        self.Q = {}
        self.episode_rewards = []
        self.success_rate = []

    def get_q(self, u, target, v):
        return self.Q.get((u, target, v), 0.0)

    def select_next(self, u, target):
        neighbors = list(self.G.neighbors(u))
        if not neighbors:
            return None
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        return max(neighbors, key=lambda v: self.get_q(u, target, v))

    def update(self, u, target, v, reward):
        old_q = self.get_q(u, target, v)
        neighbors_of_v = list(self.G.neighbors(v))
        if neighbors_of_v:
            max_next_q = max(self.get_q(v, target, w) for w in neighbors_of_v)
        else:
            max_next_q = 0.0
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.Q[(u, target, v)] = new_q

    def compute_reward(self, u, v, target, latency):
        # Base reward: penalise latency (same as before)
        reward = -latency

        # Key addition: reward getting closer to target, penalise moving away
        dist_before = self.distances[u].get(target, 999)
        dist_after  = self.distances[v].get(target, 999)
        progress = dist_before - dist_after  # +1 if closer, -1 if further

        # Bonus for reaching target
        if v == target:
            reward += 20.0
        else:
            reward += progress * 2.0  # weight progress toward goal

        return reward

    def train(self, episodes=500, steps_per_episode=30):
        nodes = list(self.G.nodes())
        print(f"Training goal-conditioned Q-learner: "
              f"{episodes} episodes, {steps_per_episode} steps each...")

        for ep in range(episodes):
            source = random.choice(nodes)
            target = random.choice(nodes)
            while target == source:
                target = random.choice(nodes)

            u = source
            total_reward = 0
            reached = False

            for step in range(steps_per_episode):
                if u == target:
                    reached = True
                    break
                v = self.select_next(u, target)
                if v is None:
                    break
                latency = measure_latency(self.G, u, v)
                reward = self.compute_reward(u, v, target, latency)
                self.update(u, target, v, reward)
                total_reward += reward
                u = v

            self.episode_rewards.append(total_reward)
            self.success_rate.append(1 if reached else 0)

            if (ep + 1) % 100 == 0:
                recent_success = np.mean(self.success_rate[-100:]) * 100
                recent_reward  = np.mean(self.episode_rewards[-100:])
                print(f"  Episode {ep+1}: avg reward = {recent_reward:.2f}, "
                      f"success rate = {recent_success:.0f}%")

            # Decay exploration over time — exploit more as learning matures
            self.epsilon = max(0.05, self.epsilon * 0.995)

    def route(self, source, target, max_steps=50):
        path = [source]
        visited = {source}
        u = source
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
        return path

def run_comparison(router, G, n_pairs=30):
    nodes = list(G.nodes())
    results = []

    for _ in range(n_pairs):
        s = random.choice(nodes)
        t = random.choice(nodes)
        if s == t:
            continue
        try:
            sp   = nx.shortest_path(G, s, t)
            qp   = router.route(s, t)
            reached = qp[-1] == t
            if not reached:
                continue

            sp_lat = sum(measure_latency(G, sp[i], sp[i+1])
                        for i in range(len(sp)-1))
            qp_lat = sum(measure_latency(G, qp[i], qp[i+1])
                        for i in range(len(qp)-1))
            results.append({
                'sp_hops': len(sp)-1,
                'qp_hops': len(qp)-1,
                'sp_lat':  sp_lat,
                'qp_lat':  qp_lat,
            })
        except nx.NetworkXNoPath:
            continue

    return results

def visualize(router, results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Learning curve with success rate
    ax = axes[0]
    window = 50
    smoothed = np.convolve(router.episode_rewards,
                           np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color='navy', label='Avg reward')
    ax.set_title('Training Curve\n(reward rising = learning)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg reward')

    ax2 = ax.twinx()
    success_smooth = np.convolve(router.success_rate,
                                 np.ones(window)/window, mode='valid')
    ax2.plot(success_smooth * 100, color='tomato',
             linestyle='--', label='Success %')
    ax2.set_ylabel('Success rate %', color='tomato')
    ax2.set_ylim(0, 100)
    ax.set_title('Training Curve\n(reward + success rate)')

    # Plot 2: Hops comparison
    ax = axes[1]
    x = range(len(results))
    ax.plot(x, [r['sp_hops'] for r in results], 'o-',
            color='gray', label='Shortest hops')
    ax.plot(x, [r['qp_hops'] for r in results], 's-',
            color='navy', label='Q-learned hops')
    ax.set_title('Hops: Shortest vs Q-learned\n(Q may use more hops for lower latency)')
    ax.set_xlabel('Sample pair')
    ax.set_ylabel('Hops')
    ax.legend()

    # Plot 3: Latency comparison — this is the real test
    ax = axes[2]
    ax.plot(x, [r['sp_lat'] for r in results], 'o-',
            color='gray', label='Hop-shortest latency')
    ax.plot(x, [r['qp_lat'] for r in results], 's-',
            color='navy', label='Q-learned latency')
    avg_improvement = np.mean(
        [(r['sp_lat'] - r['qp_lat']) / r['sp_lat'] * 100 for r in results]
    )
    ax.set_title(f'Latency Comparison\nQ-learned avg improvement: '
                 f'{avg_improvement:.1f}%')
    ax.set_xlabel('Sample pair')
    ax.set_ylabel('Total path latency')
    ax.legend()

    plt.tight_layout()
    plt.savefig('qlearning_v2_results.png', dpi=150)
    plt.show()

    print(f"\nResults across {len(results)} successful routes:")
    print(f"  Avg hops  — shortest: {np.mean([r['sp_hops'] for r in results]):.1f}"
          f"  Q-learned: {np.mean([r['qp_hops'] for r in results]):.1f}")
    print(f"  Avg latency — shortest path: {np.mean([r['sp_lat'] for r in results]):.2f}"
          f"  Q-learned: {np.mean([r['qp_lat'] for r in results]):.2f}")
    print(f"  Avg latency improvement: {avg_improvement:.1f}%")

if __name__ == "__main__":
    print("Building mesh...")
    verts, faces = build_icosahedron()
    verts, faces = subdivide(verts, faces)
    verts, faces = subdivide(verts, faces)
    G = mesh_to_graph(verts, faces)
    G = assign_link_params(G)

    distances = precompute_distances(G)

    router = GoalConditionedRouter(G, distances, alpha=0.15,
                                   gamma=0.95, epsilon=0.3)
    router.train(episodes=500, steps_per_episode=30)

    print("\nRunning comparison on 30 random pairs...")
    results = run_comparison(router, G, n_pairs=30)

    visualize(router, results)
