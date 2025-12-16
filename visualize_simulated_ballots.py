import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import utils

def visualize_ballot_embeddings(
    seed_ballots,
    seed_counts,
    sim_ballots,
    sim_counts,
    all_cands,
    distance_fn,
    title,
    laplace_scale=1.0,
    random_state=0
):
    """
    Visualize seed vs simulated ballots via Laplacian Eigenmaps on a
    Laplace-kernel-weighted ballot similarity graph.

    This version MERGES duplicate ballot types across seed and simulated:
      - If a ballot type appears in both seed and simulated, it is represented
        as a single node.
      - Its total count is the sum of seed + simulated counts.
      - In the plot, it is shown ONCE using the seed color.

    Parameters
    ----------
    seed_ballots : list[list] or list[tuple]
        List of seed ballot types. Each ballot is an ordered iterable of candidate IDs.
    seed_counts : sequence[int]
        Counts for each seed ballot type (same length as seed_ballots).
    sim_ballots : list[list] or list[tuple]
        List of simulated ballot types.
    sim_counts : sequence[int]
        Counts for each simulated ballot type (same length as sim_ballots).
    distance_fn : callable
        Function distance_fn(b1, b2) -> float that returns a (nonnegative)
        distance between two ballots.
    laplace_scale : float, default=1.0
        Scale parameter λ for the Laplace kernel:
            w_ij = exp( - d_ij / λ )
    random_state : int, default=0
        Random seed (for possible future randomness).

    Returns
    -------
    embeddings : np.ndarray, shape (n_unique_ballots, 2)
        2D embedding coordinates for all unique ballot types.
    is_seed : np.ndarray, shape (n_unique_ballots,)
        Boolean mask indicating which nodes correspond to seed ballots
        (including those that are both seed and simulated).
    """

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # 1. Merge seed and simulated ballots, deduplicating ballot types
    # ------------------------------------------------------------------
    seed_ballots = [tuple(b) for b in seed_ballots]
    sim_ballots = [tuple(b) for b in sim_ballots]

    seed_counts = np.asarray(seed_counts, dtype=float)
    sim_counts = np.asarray(sim_counts, dtype=float)

    ballot_info = {}  # key: ballot tuple -> dict with total_count, is_seed, is_sim

    # Add seed ballots
    for b, c in zip(seed_ballots, seed_counts):
        if c == 0:
            continue

        if b not in ballot_info:
            ballot_info[b] = {"count": 0.0, "is_seed": False, "is_sim": False}
        ballot_info[b]["count"] += float(c)
        ballot_info[b]["is_seed"] = True

    # Add simulated ballots
    for b, c in zip(sim_ballots, sim_counts):
        if c == 0:
            continue
        if b not in ballot_info:
            ballot_info[b] = {"count": 0.0, "is_seed": False, "is_sim": False}
        ballot_info[b]["count"] += float(c)
        ballot_info[b]["is_sim"] = True

    # Convert to arrays/lists
    unique_ballots = [list(b) for b in ballot_info.keys()]
    counts = np.array([info["count"] for info in ballot_info.values()], dtype=float)
    is_seed = np.array([info["is_seed"] for info in ballot_info.values()], dtype=bool)
    # (We don't currently need is_sim, but could keep it if desired)
    n = len(unique_ballots)

    if n == 0:
        raise ValueError("No ballot types provided after merging.")

    # ------------------------------------------------------------------
    # 2. Compute pairwise distance matrix among all unique ballot types
    # ------------------------------------------------------------------
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = float(distance_fn(unique_ballots[i], unique_ballots[j], all_cands))
            D[i, j] = d_ij
            D[j, i] = d_ij

    # ------------------------------------------------------------------
    # 3. Convert distances to weights via Laplace kernel
    #     w_ij = exp( -d_ij / laplace_scale ), w_ii = 0
    # ------------------------------------------------------------------
    if laplace_scale <= 0:
        raise ValueError("laplace_scale must be positive.")

    # with np.printoptions(threshold=np.inf):
    #     print(D)
    W = np.exp(-D / laplace_scale)
    print(np.mean(W))

    np.fill_diagonal(W, 0.0)  # no self-loops

    # ------------------------------------------------------------------
    # 4. Build graph Laplacian and compute Laplacian Eigenmaps
    # ------------------------------------------------------------------
    degrees = W.sum(axis=1)
    degrees_safe = np.where(degrees > 0, degrees, 1.0)
    D_deg = np.diag(degrees_safe)

    L = D_deg - W

    eigvals, eigvecs = np.linalg.eigh(L)
    idx_sorted = np.argsort(eigvals)
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    # Skip the (approximately) constant eigenvector, take the next two
    if n < 3:
        embed_dims = min(2, n)
        X = eigvecs[:, 1:1 + embed_dims]
        if embed_dims == 1:
            X = np.hstack([X, np.zeros((n, 1))])
    else:
        X = eigvecs[:, 1:3]

    embeddings = X  # shape (n, 2)

    # ------------------------------------------------------------------
    # 5. Prepare colors with opacity based on total counts
    # ------------------------------------------------------------------
    if np.all(counts == counts[0]):
        alphas = np.ones_like(counts, dtype=float)
    else:
        c_min = counts.min()
        c_max = counts.max()
        alphas = 0.2 + 0.8 * (counts - c_min) / (c_max - c_min)

    base_seed = np.array([0.121, 0.466, 0.705])   # blue
    base_sim = np.array([0.843, 0.373, 0.000])    # orange

    colors = np.zeros((n, 4), dtype=float)
    for i in range(n):
        # If the ballot type is seed (possibly also simulated), use seed color;
        # otherwise, use simulated color.
        rgb = base_seed if is_seed[i] else base_sim
        colors[i, :3] = rgb
        colors[i, 3] = alphas[i]

    # ------------------------------------------------------------------
    # 6. Plot the embeddings
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=colors,
        s=40,
        edgecolors='none'
    )

    ax.set_xlabel("Laplacian eigenmap dim 1")
    ax.set_ylabel("Laplacian eigenmap dim 2")
    ax.set_title("Ballot-type Laplacian Eigenmap Embedding\n(Laplace kernel on distances)")

    # Legend: seed vs simulated (note merged nodes with both appear as seed)
    seed_handle = Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor=base_seed,
        markersize=8,
        label='Seed ballots (incl. shared)'
    )
    sim_handle = Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor=base_sim,
        markersize=8,
        label='Simulated-only ballots'
    )
    ax.legend(handles=[seed_handle, sim_handle], loc='best')

    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    fig.savefig(f"plots/models/{title}.pdf", bbox_inches="tight")
    return embeddings, is_seed


if __name__ == "__main__":
    burlington_filename = "data/preflib/elections-all/burlington/ED-00005-00000002.toi"
    # burlington_filename = "data/preflib/elections-all/sf/ED-00021-00000007.toi"
    
    ballots, ballot_counts, cand_names, skipped_votes = \
        utils.read_preflib(burlington_filename)
    num_ballots = np.sum(ballot_counts)

    sample_size = int(0.005 * num_ballots)
    sample_counts = utils.resample(ballot_counts, sample_size=sample_size, with_replacement=False)

    model_names = [
        "Bootstrap",
        "PL",
        "Contextual By Length",
        "Mallows Dispersion 1",
        "Contextual Perturbation Dispersion 1",
        "Uniform"
    ]

    for model_name in model_names:

        ballot_model = utils.get_model_object(model_name, cand_names)

        ballot_model.fit(ballots.copy(), sample_counts)

        simulated_ballots, simulated_counts = ballot_model.simulate_ballots(num_ballots - sample_size)
        # simulated_ballots, simulated_counts = ballot_model.simulate_ballots(1)

        visualize_ballot_embeddings(
            ballots.copy(),
            sample_counts,
            simulated_ballots.copy(),
            simulated_counts,
            cand_names,
            title=model_name,
            distance_fn=utils.footrule_distance_partial,
            laplace_scale=20
            )