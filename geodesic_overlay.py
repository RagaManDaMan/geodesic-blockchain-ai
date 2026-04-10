import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx

# --- Step 1: Build a base icosahedron from scratch ---
# An icosahedron has 12 vertices, 20 faces, 30 edges.
# This is the geometric foundation the patent describes.

def build_icosahedron():
    phi = (1 + np.sqrt(5)) / 2  # golden ratio ~1.618
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=float)
    # Normalize to unit sphere
    verts /= np.linalg.norm(verts[0])

    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]
    return verts, faces

# --- Step 2: Subdivide each triangle into 4 smaller triangles ---
# This is exactly what the patent means by "recursive subdivision"

def subdivide(verts, faces):
    verts = list(verts)
    new_faces = []
    midpoint_cache = {}

    def midpoint(i, j):
        key = tuple(sorted([i, j]))
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid = (np.array(verts[i]) + np.array(verts[j])) / 2
        mid /= np.linalg.norm(mid)  # project back onto unit sphere
        idx = len(verts)
        verts.append(mid)
        midpoint_cache[key] = idx
        return idx

    for f in faces:
        a, b, c = f
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]

    return np.array(verts), new_faces

# --- Step 3: Build a graph from the mesh ---
# Each vertex is a potential blockchain node.
# Each edge is a potential network link with a weight.

def mesh_to_graph(verts, faces):
    G = nx.Graph()
    for i, v in enumerate(verts):
        G.add_node(i, pos=v)
    for f in faces:
        for i in range(3):
            u, v = f[i], f[(i+1)%3]
            if not G.has_edge(u, v):
                dist = np.linalg.norm(verts[u] - verts[v])
                G.add_edge(u, v, weight=round(dist, 4))
    return G

# --- Step 4: Visualize ---

def visualize(verts, faces, G):
    fig = plt.figure(figsize=(14, 6))

    # Left: 3D mesh
    ax1 = fig.add_subplot(121, projection='3d')
    poly = Poly3DCollection(
        [[verts[f[0]], verts[f[1]], verts[f[2]]] for f in faces],
        alpha=0.15, edgecolor='steelblue', facecolor='lightcyan'
    )
    ax1.add_collection3d(poly)
    ax1.scatter(*verts.T, color='navy', s=10)
    ax1.set_title(f'Geodesic Mesh\n{len(verts)} nodes, {len(faces)} faces')
    ax1.set_box_aspect([1,1,1])
    ax1.axis('off')

    # Right: degree distribution (how many edges per node)
    ax2 = fig.add_subplot(122)
    degrees = [d for _, d in G.degree()]
    ax2.hist(degrees, bins=range(min(degrees), max(degrees)+2),
             color='steelblue', edgecolor='white', align='left')
    ax2.set_title('Node Degree Distribution\n(edges per blockchain node)')
    ax2.set_xlabel('Degree')
    ax2.set_ylabel('Count')
    ax2.axvline(np.mean(degrees), color='red', linestyle='--',
                label=f'Mean: {np.mean(degrees):.1f}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('geodesic_mesh.png', dpi=150)
    plt.show()
    print(f"\nMesh saved to geodesic_mesh.png")

# --- Run it ---
if __name__ == "__main__":
    print("Building icosahedron...")
    verts, faces = build_icosahedron()
    print(f"  Base: {len(verts)} vertices, {len(faces)} faces")

    print("Subdividing (1 level)...")
    verts, faces = subdivide(verts, faces)
    print(f"  After 1 subdivision: {len(verts)} vertices, {len(faces)} faces")

    print("Subdividing (2 levels)...")
    verts, faces = subdivide(verts, faces)
    print(f"  After 2 subdivisions: {len(verts)} vertices, {len(faces)} faces")

    print("Building network graph...")
    G = mesh_to_graph(verts, faces)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Avg degree: {np.mean([d for _,d in G.degree()]):.1f}")
    print(f"  Diameter (max hops): {nx.diameter(G)}")

    print("\nVisualizing...")
    visualize(verts, faces, G)
