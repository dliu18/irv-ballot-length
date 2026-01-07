import matplotlib.pyplot as plt
from collections import Counter

import utils
import numpy as np 

from math import sqrt

import pickle

def visualize_ballot_tree(
    cand_names,
    ballots,
    ballot_counts,
    title="Ballot Tree",
    filename="ballot_tree",
    figsize=(20, 7),
    node_size=850,
    min_alpha=0.03,
    zero_alpha=0.03,
    aggregate_prefixes=False,
    truncate_full_rankings=True,
    vertical_spacing=1.8,
    ax=None,
    *,
    # If provided, overrides computed node weights for prefixes.
    node_weights_override=None,
    # If True, interpret node_weights as signed errors and color accordingly.
    error_mode=False,
    # Normalization constant for error-mode alpha scaling. Alpha is min(1, abs(err)/error_alpha_n).
    error_alpha_n=None,
):
    """
    Visualize the permutation tree of all possible ballots over the candidates in `cand_names`.

    Each node is a prefix (partial ranking). The node's weight is determined by either:
      1) Exact ballots only (aggregate_prefixes=False): node count = # ballots exactly equal to that prefix
      2) Prefix aggregation (aggregate_prefixes=True): node count = # ballots that start with that prefix

    If truncate_full_rankings=True, then any ballot that ranks *all* candidates (len(ballot)==m) is first
    truncated to length m-1 by dropping the final candidate. (In full rankings, the last candidate is
    determined by the others and typically not informative in a choice-by-stage view.)

    If error_mode=True, node weights are treated as signed errors:
        error(prefix) = true(prefix) - estimated(prefix)
    Nodes are colored green for positive and red for negative. Opacity scales linearly with abs(error),
    reaching 1.0 when abs(error) == error_alpha_n.

    Returns (fig, ax).
    """
    if len(ballots) != len(ballot_counts):
        raise ValueError("ballots and ballot_counts must have the same length.")

    cands = list(cand_names.keys())
    cand_set = set(cands)
    m = len(cands)

    def _normalize_ballot(b):
        b = tuple(b)
        if len(b) == 0:
            raise ValueError("Encountered an empty (length 0) ballot.")
        if not set(b).issubset(cand_set):
            unknown = set(b) - cand_set
            raise ValueError(f"Ballot {list(b)} includes unknown candidates: {sorted(unknown)}")
        if truncate_full_rankings and len(b) == m:
            # drop the final candidate (forced by the others in a full ranking)
            b = b[:-1]
        return b

    # --- Compute node weights (counts) unless explicitly overridden ---
    node_weights = Counter()
    if node_weights_override is None:
        for ballot, count in zip(ballots, ballot_counts):
            b = _normalize_ballot(ballot)
            if aggregate_prefixes:
                for L in range(1, len(b) + 1):
                    node_weights[b[:L]] += int(count)
            else:
                node_weights[b] += int(count)
    else:
        node_weights.update(node_weights_override)

    # --- Build permutation tree (optionally stopping at depth m-1) ---
    max_depth = (m - 1) if truncate_full_rankings else m
    nodes = {}  # prefix -> {"children": [], "depth": int}

    def build_tree(prefix):
        depth = len(prefix)
        nodes[prefix] = nodes.get(prefix, {"children": [], "depth": depth})
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

    # --- Assign coordinates (x by recursive centering, y by depth) ---
    x_pos = {}
    y_pos = {}
    leaf_counter = [0]

    def assign_positions(prefix):
        depth = nodes[prefix]["depth"]
        y_pos[prefix] = -depth * float(vertical_spacing)

        children = nodes[prefix]["children"]
        if not children:
            x_pos[prefix] = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            child_xs = []
            for child in children:
                assign_positions(child)
                child_xs.append(x_pos[child])
            x_pos[prefix] = sum(child_xs) / len(child_xs)

    assign_positions(root)

    # --- Visual style helpers ---
    if error_mode:
        if error_alpha_n is None:
            raise ValueError("When error_mode=True, you must provide error_alpha_n (typically total ballots n).")

        def facecolor_for(prefix):
            if prefix == root:
                return "white"
            val = float(node_weights.get(prefix, 0.0))
            if val > 0:
                return "green"
            if val < 0:
                return "red"
            return "white"

        def alpha_for(prefix):
            if prefix == root:
                return 0.0
            err = float(node_weights.get(prefix, 0.0))
            a = min(1.0, abs(err) / float(error_alpha_n))
            return a

        def label_text_for(prefix):
            if prefix == root:
                return ""
            val = node_weights.get(prefix, 0)
            return f"{int(val):+d}" if val != 0 else "0"
    else:
        nonzero = [v for v in node_weights.values() if v > 0]
        max_count = max(nonzero) if nonzero else 1

        def facecolor_for(prefix):
            return "black" if prefix != root else "white"

        def alpha_for(prefix):
            if prefix == root:
                return 0.0
            count = node_weights.get(prefix, 0)
            if count == 0:
                return zero_alpha
            return max(min_alpha, count / max_count)

        def label_text_for(prefix):
            if prefix == root:
                return ""
            count = node_weights.get(prefix, 0)
            return str(int(count)) if count > 0 else ""

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Prevent vertical compression: set explicit axis limits based on tree depth.
    y_pad = float(vertical_spacing)
    ax.set_ylim(-(max_depth + 1) * y_pad, y_pad)
    if leaf_counter[0] > 0:
        x_pad = 0.6
        ax.set_xlim(-x_pad, (leaf_counter[0] - 1) + x_pad)


    # --- Draw edges ---
    for parent, info in nodes.items():
        for child in info["children"]:
            ax.plot(
                [x_pos[parent], x_pos[child]],
                [y_pos[parent], y_pos[child]],
                color="gray",
                linewidth=1,
                alpha=0.7,
                zorder=1,
            )

    # --- Draw nodes ---
    for prefix in nodes.keys():
        x = x_pos[prefix]
        y = y_pos[prefix]
        alpha = alpha_for(prefix)

        # Root: skip drawing circle, just show label if desired
        if prefix == root:
            continue

        count_or_err = node_weights.get(prefix, 0)
        ax.scatter(
            x,
            y,
            s=node_size,
            c=facecolor_for(prefix),
            alpha=alpha,
            edgecolors="black",
            linewidths=1,
            zorder=2,
        )

        # Candidate label ABOVE circle
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
            zorder=3,
        )

        # Value INSIDE circle
        text_inside = label_text_for(prefix)
        if text_inside != "":
            # In error mode, use black text (colors convey sign).
            if error_mode:
                text_color = "black"
            else:
                text_color = "white" if alpha > 0.35 else "black"
            ax.text(
                x,
                y,
                text_inside,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                zorder=4,
            )

    # --- Formatting ---
    ax.set_title(title, fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    if filename is not None:
        plt.tight_layout()
        fig.savefig(f"plots/ballot-trees/{filename}.pdf", bbox_inches="tight")

    return fig, ax


def visualize_estimated_and_error_trees(
    cand_names,
    true_ballots,
    true_counts,
    est_ballots,
    est_counts,
    *,
    error_alpha_n=None,
    aggregate_prefixes=False,
    truncate_full_rankings=True,
    vertical_spacing=1.8,
    figsize=(20, 14),
    node_size=850,
    filename="ballot_tree_with_error",
    title_left="Estimated distribution",
    title_right="True - Estimated (error)",
):
    """
    Create a single figure with two subplots:
      (left) estimated ballot tree
      (right) error tree where node value = true - estimated

    Error tree nodes are colored green (positive) and red (negative), with opacity
    alpha = min(1, abs(error)/n) where n is the total number of ballots in the true distribution.
    """
    n = int(np.sum(true_counts))

    # Choose a taller figure for deeper trees to avoid vertical constriction.
    m = len(cand_names)
    max_depth = (m - 1) if truncate_full_rankings else m
    # figsize = (
    #     float(figsize[0]),
    #     max(float(figsize[1]), 1.6 * max_depth * float(vertical_spacing)),
    # )


    # First compute prefix weights for true and estimated using the same options.
    def compute_prefix_weights(ballots, counts):
        tmp = Counter()
        # Reuse visualize_ballot_tree's normalization logic by calling it in "dry" mode
        # via its node_weights_override: we implement a tiny local version here for speed/clarity.
        cands = list(cand_names.keys())
        cand_set = set(cands)
        m = len(cands)

        def _normalize_ballot(b):
            b = tuple(b)
            if len(b) == 0:
                raise ValueError("Encountered an empty (length 0) ballot.")
            if not set(b).issubset(cand_set):
                unknown = set(b) - cand_set
                raise ValueError(f"Ballot {list(b)} includes unknown candidates: {sorted(unknown)}")
            if truncate_full_rankings and len(b) == m:
                b = b[:-1]
            return b

        for ballot, count in zip(ballots, counts):
            b = _normalize_ballot(ballot)
            if aggregate_prefixes:
                for L in range(1, len(b) + 1):
                    tmp[b[:L]] += int(count)
            else:
                tmp[b] += int(count)
        return tmp

    true_w = compute_prefix_weights(true_ballots, true_counts)
    est_w = compute_prefix_weights(est_ballots, est_counts)

    # Signed error: estimated - true
    err_w = Counter()
    keys = set(true_w.keys()) | set(est_w.keys())
    for k in keys:
        err_w[k] = int(est_w.get(k, 0)) - int(true_w.get(k, 0))

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    visualize_ballot_tree(
        cand_names,
        est_ballots,
        est_counts,
        title=title_left,
        filename=None,
        figsize=figsize,
        node_size=node_size,
        aggregate_prefixes=aggregate_prefixes,
        truncate_full_rankings=truncate_full_rankings,
        vertical_spacing=vertical_spacing,
        ax=axes[0],
    )

    visualize_ballot_tree(
        cand_names,
        est_ballots,  # unused when node_weights_override is provided
        est_counts,
        title=title_right,
        filename=None,
        figsize=figsize,
        node_size=node_size,
        aggregate_prefixes=aggregate_prefixes,
        truncate_full_rankings=truncate_full_rankings,
        vertical_spacing=vertical_spacing,
        ax=axes[1],
        node_weights_override=err_w,
        error_mode=True,
        error_alpha_n=error_alpha_n,
    )

    plt.tight_layout()
    fig.savefig(f"plots/ballot-trees/{filename}.pdf", bbox_inches="tight")
    return fig, axes

if __name__ == "__main__":
    burlington_filename = "data/preflib/elections-all/burlington/ED-00005-00000002.toi"
    
    ballots, ballot_counts, cand_names, skipped_votes = \
        utils.read_preflib(burlington_filename)

    filtered_cands = [1, 2, 4, 5]
    filtered_cand_names = {cand: full_name.split(" ")[1] for cand, full_name in cand_names.items() if cand in filtered_cands}
    filtered_ballots, filtered_ballot_counts = utils.reduce_election(ballots, ballot_counts, filtered_cands)
    n = np.sum(filtered_ballot_counts)

    ctr = Counter()
    for ballot_idx, ballot in enumerate(filtered_ballots):
        top_choice = ballot[0]
        ctr[top_choice] += filtered_ballot_counts[ballot_idx]
    print(ctr)
    
    visualize_ballot_tree(
        filtered_cand_names, 
        filtered_ballots, 
        filtered_ballot_counts,
        aggregate_prefixes=False, 
        title="Burlington True Election Profile",
        filename="burlington_election_profile")

    pickle_filename = "results/push_pull_eval/inferred_distribution_by_election_filtered_burlington-02.pickle"
    with open(pickle_filename, "rb") as picklefile:
        inferred_distributions = pickle.load(picklefile)

    
    for model_name in inferred_distributions["burlington-02"]:
        D_hat = inferred_distributions["burlington-02"][model_name][50][0]
        simulated_ballots, simulated_counts = utils.get_ballot_sample_from_distribution(D_hat, n, seed=0)

        est_ballots, est_counts = utils.reduce_election(simulated_ballots, simulated_counts, filtered_cands)

        visualize_estimated_and_error_trees(
            filtered_cand_names,
            true_ballots=filtered_ballots,
            true_counts=filtered_ballot_counts,
            est_ballots=est_ballots,
            est_counts=est_counts,
            error_alpha_n=np.max(ballot_counts)/2,
            aggregate_prefixes=False,
            truncate_full_rankings=True,
            filename="burlington_" + "_".join(model_name.lower().split(" ")) + "_with_error",
            title_left=f"Burlington Estimated Election Profile ({model_name})",
            title_right="Estimated - True (error)",
        )
