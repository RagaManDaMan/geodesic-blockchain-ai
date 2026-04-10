import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from geodesic_overlay import build_icosahedron, subdivide, mesh_to_graph

# --- What this does ---
# Each edge in the geodesic mesh has an assigned L (inductance) and C (capacitance)
# borrowed from Telegrapher's equations — used here as a proxy for link quality.
# The Q-learner walks the graph, measures "latency" on each hop,
# and learns which edges are fastest. Over time it builds a routing table.

random.seed(42)
np.random.seed(42)

# --- Assign fake-but-realistic L, C values to each edge ---
# In a real system these would come from actual network measurements.
# Here we simulate: some edges are fast (low LC), some are slow (high LC).

def assign_link_params(G):
    for u, v in G.edges():
        G[u][v]['L'] = np.random.uniform(0.1, 1.0)   # inductance proxy
        G[u][v]['C'] = np.random.uniform(0.1, 1.0)   # capacitance proxy
        # Telegrapher's delay: v = 1 / sqrt(L * C)
        # Higher = slower link. We use it as latency cost.
        G[u][v]['delay'] = 1.0 / np.sqrt(G[u][v]['L'] * G[u][v]['C'])
        # Add some noise to simulate real network jitter
        G[u][v]['base_latency'] = G[u][v]['delay'] + np.random.uniform(0, 0.5)
    return G

def measure_latency(G, u, v):
    # Simulates a real latency measurement with jitter
    base = G[u][v]['base_latency']
    jitter = np.random.normal(0, 0.05)
    return max(0.01, base + jitter)

# --- Q-learning agent ---

class QLearningRouter:
    def __init__(self, G, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.G = G
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.Q = {}             # Q[(u,v)] = estimated value of using edge u->v
        self.episode_rewards = []

    def get_q(self, u, v):
        return self.Q.get((u, v), 0.0)

    def select_next(self, u):
        neighbors = list(self.G.neighbors(u))
        if not neighbors:
            return None
        # Explore: pick random neighbor
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        # Exploit: pick neighbor with best Q value
        return max(neighbors, key=lambda v: self.get_q(u, v))

    def update(self, u, v, reward):
        old_q = self.get_q(u, v)
        neighbors_of_v = list(self.G.neighbors(v))
        if neighbors_of_v:
            max_next_q = max(self.get_q(v, w) for w in neighbors_of_v)
        else:
            max_next_q = 0.0
        # Bellman equation
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.Q[(u, v)] = new_q

    def train(self, episodes=300, steps_per_episode=40):
        nodes = list(self.G.nodes())
        print(f"\nTraining Q-learner: {episodes} episodes, "
              f"{steps_per_episode} steps each...")

        for ep in range(episodes):
            u = random.choice(nodes)
            total_reward = 0
            for _ in range(steps_per_episode):
                v = self.select_next(u)
                if v is None:
                    break
                latency = measure_latency(self.G, u, v)
                reward = -latency   # negative: lower latency = better reward
                self.update(u, v, reward)
                total_reward += reward
                u = v
            self.episode_rewards.append(total_reward)

            if (ep + 1) % 50 == 0:
                print(f"  Episode {ep+1}: avg reward = "
                      f"{np.mean(self.episode_rewards[-50:]):.3f}")

# --- Use learned Q-values to find best path between two nodes ---

def best_path(router, source, target, max_steps=50):
    # Greedy walk using learned Q-values (no exploration)
    path = [source]
    visited = {source}
    u = source
    for _ in range(max_steps):
        if u == target:
            break
        neighbors = [n for n in router.G.neighbors(u) if n not in visited]
        if not neighbors:
            break
        v = max(neighbors, key=lambda n: router.get_q(u, n))
        path.append(v)
        visited.add(v)
        u = v
    return path

# --- Visualize training progress and edge weight distribution ---

def visualize_results(router, G):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Learning curve
    ax = axes[0]
    window = 20
    smoothed = np.convolve(router.episode_rewards,
                           np.ones(window)/window, mode='valid')
    ax.plot(router.episode_rewards, alpha=0.3, color='steelblue', label='Raw')
    ax.plot(smoothed, color='navy', label=f'{window}-ep average')
    ax.set_title('Q-Learner Training Curve\n(reward should rise over time)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()

    # Plot 2: Distribution of learned Q-values
    ax = axes[1]
    q_vals = list(router.Q.values())
    ax.hist(q_vals, bins=40, color='steelblue', edgecolor='white')
    ax.set_title('Learned Q-Value Distribution\n(higher = preferred edge)')
    ax.set_xlabel('Q-value')
    ax.set_ylabel('Count')

    # Plot 3: Compare shortest path (hops) vs Q-learned path (latency)
    ax = axes[2]
    nodes = list(G.nodes())
    sample_pairs = [(random.choice(nodes), random.choice(nodes))
                    for _ in range(30)]
    sample_pairs = [(s, t) for s, t in sample_pairs if s != t]

    hop_counts = []
    q_path_latencies = []
    shortest_latencies = []

    for s, t in sample_pairs[:20]:
        # Shortest path by hops (what a naive network does)
        try:
            sp = nx.shortest_path(G, s, t)
            hop_lat = sum(measure_latency(G, sp[i], sp[i+1])
                         for i in range(len(sp)-1))
            shortest_latencies.append(hop_lat)

            # Q-learned path
            qp = best_path(router, s, t)
            if qp[-1] == t:
                q_lat = sum(measure_latency(G, qp[i], qp[i+1])
                           for i in range(len(qp)-1))
                q_path_latencies.append(q_lat)
                hop_counts.append(len(sp) - 1)
        except nx.NetworkXNoPath:
            continue

    x = range(len(q_path_latencies))
    ax.plot(x, shortest_latencies[:len(x)], 'o-',
            color='gray', label='Shortest hops path')
    ax.plot(x, q_path_latencies, 's-',
            color='navy', label='Q-learned path')
    ax.set_title('Path Latency: Hop-shortest vs Q-learned\n(lower = better)')
    ax.set_xlabel('Sample pair')
    ax.set_ylabel('Total latency')
    ax.legend()

    plt.tight_layout()
    plt.savefig('qlearning_results.png', dpi=150)
    plt.show()
    print("\nResults saved to qlearning_results.png")

# --- Run it ---
if __name__ == "__main__":
    print("Building geodesic mesh (2 subdivisions = 162 nodes)...")
    verts, faces = build_icosahedron()
    verts, faces = subdivide(verts, faces)
    verts, faces = subdivide(verts, faces)
    G = mesh_to_graph(verts, faces)

    print("Assigning link parameters (L, C, latency)...")
    G = assign_link_params(G)

    print("Sample edge delays (first 5):")
    for u, v, d in list(G.edges(data=True))[:5]:
        print(f"  Edge ({u},{v}): delay={d['delay']:.3f}, "
              f"base_latency={d['base_latency']:.3f}")

    router = QLearningRouter(G, alpha=0.1, gamma=0.9, epsilon=0.2)
    router.train(episodes=300, steps_per_episode=40)

    print("\nTesting a sample route (node 0 -> node 80)...")
    path = best_path(router, 0, 80)
    print(f"  Q-learned path: {len(path)} hops")
    sp = nx.shortest_path(G, 0, 80)
    print(f"  Shortest path:  {len(sp)} hops")
    print(f"  Q-path found target: {path[-1] == 80}")

    visualize_results(router, G)
