import matplotlib.pyplot as plt
from collections import Counter

import utils
import numpy as np 

from math import sqrt

def visualize_ballot_tree(
    cand_names,
    ballots,
    ballot_counts,
    title="Ballot Tree",
    figsize=(20, 7),
    node_size=850,
    min_alpha=0.15,
    zero_alpha=0.03,
    aggregate_prefixes=False,
    truncate_full_rankings=True,
):
    """
    Visualize the permutation tree of all possible ballots over the candidates in `cand_names`.

    Each node is a prefix (partial ranking). The node's opacity and count are determined by either:
      1) Exact ballots only (aggregate_prefixes=False): node count = # ballots exactly equal to that prefix
      2) Prefix aggregation (aggregate_prefixes=True): node count = # ballots that start with that prefix

    If truncate_full_rankings=True, then any ballot that ranks *all* candidates (len(ballot)==m) is first
    truncated to length m-1 by dropping the final candidate. (In full rankings, the last candidate is
    determined by the others and typically not informative in a choice-by-stage view.)

    Additional features:
    - Candidate label placed above each circle.
    - Ballot count placed inside the circle.
    """

    # --- Validation ---
    if len(ballots) != len(ballot_counts):
        raise ValueError("ballots and ballot_counts must have the same length.")

    cands = list(cand_names.keys())
    cand_set = set(cands)
    m = len(cands)

    def _normalize_ballot(b):
        # Accept list/tuple/np array; return tuple
        b = tuple(b)
        if len(b) == 0:
            raise ValueError("Encountered an empty (length 0) ballot.")
        if not set(b).issubset(cand_set):
            unknown = set(b) - cand_set
            raise ValueError(f"Ballot {list(b)} includes unknown candidates: {sorted(unknown)}")
        if truncate_full_rankings and len(b) == m:
            # Drop the final candidate for full rankings
            b = b[:-1]
        return b

    # --- Count ballots into node weights ---
    node_weights = Counter()
    for ballot, count in zip(ballots, ballot_counts):
        b = _normalize_ballot(ballot)
        if aggregate_prefixes:
            for L in range(1, len(b) + 1):
                node_weights[b[:L]] += count
        else:
            node_weights[b] += count

    # --- Build permutation tree (optionally stopping at depth m-1) ---
    max_depth = (m - 1) if truncate_full_rankings else m

    nodes = {}  # prefix -> {"children": [], "depth": int}

    def build_tree(prefix):
        depth = len(prefix)
        nodes[prefix] = nodes.get(prefix, {"children": [], "depth": depth})

        # Stop early if we're truncating full rankings (so we don't draw the final forced candidate)
        if depth >= max_depth:
            return

        remaining = [c for c in cands if c not in prefix]
        if not remaining:
            return  # leaf

        for c in remaining:
            child = prefix + (c,)
            nodes[prefix]["children"].append(child)
            build_tree(child)

    root = tuple()
    build_tree(root)

    # --- Assign hierarchical layout (X = leaf order, Y = depth) ---
    x_pos = {}
    y_pos = {}
    next_x = 0

    def assign_positions(prefix):
        nonlocal next_x
        children = nodes[prefix]["children"]
        y_pos[prefix] = -nodes[prefix]["depth"]

        if not children:
            x_pos[prefix] = next_x
            next_x += 1
        else:
            child_xs = []
            for child in children:
                assign_positions(child)
                child_xs.append(x_pos[child])
            x_pos[prefix] = sum(child_xs) / len(child_xs)

    assign_positions(root)

    # --- Compute opacity ---
    nonzero = [v for v in node_weights.values() if v > 0]
    max_count = max(nonzero) if nonzero else 1

    def alpha_for(prefix):
        if prefix == root:
            return 0.0
        count = node_weights.get(prefix, 0)
        if count == 0:
            return zero_alpha
        # sqrt scaling helps compress large ranges while keeping small differences visible
        return max(min_alpha, sqrt(count / max_count))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    for prefix, info in nodes.items():
        for child in info["children"]:
            ax.plot(
                [x_pos[prefix], x_pos[child]],
                [y_pos[prefix], y_pos[child]],
                color="gray",
                linewidth=1,
                alpha=0.5,
            )

    # Draw nodes + labels
    for prefix in nodes:
        x, y = x_pos[prefix], y_pos[prefix]
        alpha = alpha_for(prefix)
        count = node_weights.get(prefix, 0)

        ax.scatter(
            x,
            y,
            s=node_size,
            color="tab:blue",
            alpha=alpha,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

        # Candidate label ABOVE circle
        if prefix != root:
            label = str(cand_names[prefix[-1]])
            ax.text(
                x,
                y + 0.15,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="black",
            )

        # Count INSIDE circle
        if count > 0:
            ax.text(
                x,
                y,
                str(count),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if alpha > 0.35 else "black",
                zorder=4,
            )

    # --- Formatting ---
    ax.set_title(title, fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(f"plots/ballot-trees/{title}.pdf", bbox_inches="tight")

if __name__ == "__main__":
    burlington_filename = "data/preflib/elections-all/burlington/ED-00005-00000002.toi"
    
    ballots, ballot_counts, cand_names, skipped_votes = \
        utils.read_preflib(burlington_filename)

    filtered_cands = [1, 2, 4, 5]
    filtered_cand_names = {cand: full_name.split(" ")[1] for cand, full_name in cand_names.items() if cand in filtered_cands}
    filtered_ballots, filtered_ballot_counts = utils.reduce_election(ballots, ballot_counts, filtered_cands)

    ctr = Counter()
    for ballot_idx, ballot in enumerate(filtered_ballots):
        top_choice = ballot[0]
        ctr[top_choice] += filtered_ballot_counts[ballot_idx]
    print(ctr)
    
    visualize_ballot_tree(
        filtered_cand_names, 
        filtered_ballots, 
        filtered_ballot_counts,
        aggregate_prefixes=True, 
        title="burlington-agg")
