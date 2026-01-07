from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

import utils


# ============================================================
# Candidate space
# ============================================================

@dataclass(frozen=True)
class CandidateSpace:
    candidates: Tuple[int, ...]

    def __post_init__(self):
        if len(self.candidates) == 0:
            raise ValueError("candidates cannot be empty.")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidates must be unique.")

    @property
    def m(self) -> int:
        return len(self.candidates)

    @property
    def id_to_idx(self) -> Dict[int, int]:
        return {cid: i for i, cid in enumerate(self.candidates)}

    @property
    def idx_to_id(self) -> Dict[int, int]:
        return {i: cid for i, cid in enumerate(self.candidates)}

    def encode_ranking(self, ranking: Sequence[int]) -> List[int]:
        id2i = self.id_to_idx
        out: List[int] = []
        seen = set()
        for cid in ranking:
            if cid not in id2i:
                raise ValueError(f"Candidate {cid} not in candidate set.")
            if cid in seen:
                raise ValueError(f"Repeated candidate {cid} in ranking {ranking}.")
            seen.add(cid)
            out.append(id2i[cid])
        return out


# ============================================================
# Interaction choice model
# ============================================================

class InteractionChoiceModel(nn.Module):
    """
    V(j) = theta_j + (1/|A|) * sum_{i in A} u_{i,j}
    P(j | A) = softmax(V(j)) over remaining candidates + END
    """

    def __init__(
        self,
        candidate_space: CandidateSpace,
        *,
        init_scale: float = 0.01,
        context_effects: bool = True,
        l2_lambda: float = 0.0,
        laplacian_lambda: float = 0.0,
        rank_heterogeneous: bool = False,
        skip_zeros: bool = False
    ):
        super().__init__()
        self.space = candidate_space
        self.m = candidate_space.m
        self.end_action = self.m
        self.num_actions = self.m + 1
        self.skip_zeros = skip_zeros

        self.context_effects = context_effects

        # Regularization coefficients
        self.l2_lambda = float(l2_lambda)
        self.laplacian_lambda = float(laplacian_lambda)

        self.rank_heterogeneous = rank_heterogeneous
        self.num_ranks = self.m if rank_heterogeneous else 1

        self.theta = nn.Parameter(
            init_scale * torch.randn(self.num_ranks, self.num_actions)
        )
        self.U = nn.Parameter(
            init_scale * torch.randn(self.num_ranks, self.m, self.num_actions)
        )

    # ---------------- internal helpers ----------------

    def _rank_index(self, prefix_len: int) -> int:
        return 0 if not self.rank_heterogeneous else min(prefix_len, self.num_ranks - 1)

    def _state_from_prefix_idx(self, prefix_idx: Sequence[int]) -> int:
        state = 0
        for a in prefix_idx:
            state |= (1 << int(a))
        return state

    def _available_mask(self, state: int, device) -> torch.Tensor:
        mask = torch.ones(self.num_actions, dtype=torch.bool, device=device)
        for j in range(self.m):
            if (state >> j) & 1:
                mask[j] = False
        mask[self.end_action] = True
        return mask

    def _values_for_state(self, state: int, prefix_len: int) -> torch.Tensor:
        r = self._rank_index(prefix_len)
        V = self.theta[r].clone()

        if self.context_effects:
            chosen = [i for i in range(self.m) if (state >> i) & 1]
            if chosen:
                idx = torch.tensor(chosen, device=V.device)
                V = V + self.U[r].index_select(0, idx).mean(dim=0)

        return V

    def _log_probs_at_state(self, state: int, prefix_len: int) -> torch.Tensor:
        V = self._values_for_state(state, prefix_len)
        avail = self._available_mask(state, V.device)
        return F.log_softmax(V.masked_fill(~avail, float("-inf")), dim=-1)

    def _probs_at_state(self, state: int, prefix_len: int) -> torch.Tensor:
        V = self._values_for_state(state, prefix_len)
        avail = self._available_mask(state, V.device)
        return torch.softmax(V.masked_fill(~avail, float("-inf")), dim=-1)

    # ---------------- likelihood ----------------

    def log_prob_ranking_idx(self, ranking_idx: Sequence[int]) -> torch.Tensor:
        device = self.theta.device
        state = 0
        logp = torch.tensor(0.0, device=device)

        for t, a in enumerate(ranking_idx):
            if (state >> a) & 1:
                raise ValueError("Repeated candidate in ranking.")
            logp = logp + self._log_probs_at_state(state, t)[a]
            state |= (1 << a)

        logp = logp + self._log_probs_at_state(state, len(ranking_idx))[self.end_action]
        return logp

    def nll_from_counts(
        self,
        rankings: List[Sequence[int]],
        counts: Sequence[int],
    ) -> torch.Tensor:
        device = self.theta.device
        counts_t = torch.tensor(counts, dtype=torch.float32, device=device)
        total = counts_t.sum().clamp_min(1.0)

        logps = []
        for idx, r in enumerate(rankings):
            idx = self.space.encode_ranking(r)
            logps.append(self.log_prob_ranking_idx(idx))
        logps = torch.stack(logps)

        base_loss = -(counts_t * logps).sum() / total

        reg = torch.tensor(0.0, device=device)

        # (1) L2 regularization on parameters
        if self.l2_lambda != 0.0:
            reg = reg + self.l2_lambda * (self.theta.pow(2).sum() + self.U.pow(2).sum())

        # (2) Smoothness across neighboring ranks (only relevant if rank-heterogeneous)
        if self.laplacian_lambda != 0.0 and self.num_ranks > 1:
            dtheta = self.theta[1:] - self.theta[:-1]
            dU = self.U[1:] - self.U[:-1]
            reg = reg + self.laplacian_lambda * (dtheta.pow(2).sum() + dU.pow(2).sum())

        return base_loss + reg

    # ---------------- public API ----------------

    def prob_ranking(self, ranking: Sequence[int]) -> float:
        idx = self.space.encode_ranking(ranking)
        with torch.no_grad():
            return float(torch.exp(self.log_prob_ranking_idx(idx)).cpu())

    def log_prob_ranking(self, ranking: Sequence[int]) -> float:
        idx = self.space.encode_ranking(ranking)
        with torch.no_grad():
            return float(self.log_prob_ranking_idx(idx).cpu())

    @torch.no_grad()
    def display_parameters(
        self,
        rank: Optional[int] = None,
        precision: int = 4,
        heatmap_path: str = "plots/U_matrices_heatmaps.pdf",
        figsize_per_rank: Tuple[float, float] = (6.0, 5.0),
        cmap: str = "coolwarm",
    ) -> None:
        """
        Display learned parameters.

        - Prints fixed effects theta to stdout.
        - Plots interaction matrices U as annotated heatmaps.
        - Saves ALL heatmaps (one per rank) into a single file.
        """
        from matplotlib.backends.backend_pdf import PdfPages

        idx_to_id = self.space.idx_to_id
        m = self.m

        # Determine which ranks to show
        if not self.rank_heterogeneous:
            ranks = [0]
        else:
            if rank is not None:
                if rank < 0 or rank >= self.num_ranks:
                    raise ValueError(f"rank must be in [0, {self.num_ranks - 1}]")
                ranks = [rank]
            else:
                ranks = list(range(self.num_ranks))

        # ---------- Print theta ----------
        for r in ranks:
            print("=" * 80)
            if self.rank_heterogeneous:
                print(f"Rank {r} (prefix length = {r})")
            else:
                print("Parameters (rank-homogeneous)")

            print("theta (fixed effects):")
            for j in range(m):
                print(f"  theta[{idx_to_id[j]}] = {self.theta[r, j].item():.{precision}f}")
            print(f"  theta[END] = {self.theta[r, self.end_action].item():.{precision}f}")
        print("=" * 80)

        # ---------- Plot annotated U matrices ----------
        with PdfPages(heatmap_path) as pdf:
            for r in ranks:
                U_r = self.U[r].detach().cpu().numpy()  # shape (m, m+1)

                fig, ax = plt.subplots(figsize=figsize_per_rank)
                im = ax.imshow(U_r, cmap=cmap, aspect="auto")

                # Axis labels
                col_labels = [idx_to_id[j] for j in range(m)] + ["END"]
                row_labels = [idx_to_id[i] for i in range(m)]

                ax.set_xticks(range(m + 1))
                ax.set_yticks(range(m))
                ax.set_xticklabels(col_labels, rotation=45, ha="right")
                ax.set_yticklabels(row_labels)

                ax.set_xlabel("Next choice j")
                ax.set_ylabel("Previously chosen i")

                title = (
                    f"Interaction matrix U (rank {r})"
                    if self.rank_heterogeneous
                    else "Interaction matrix U"
                )
                ax.set_title(title)

                # Colorbar
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("u_{ij}", rotation=270, labelpad=15)

                # Cell annotations
                vmax = float(np.nanmax(np.abs(U_r))) if U_r.size > 0 else 0.0
                threshold = 0.5 * vmax if vmax > 0 else 0.0

                for i in range(U_r.shape[0]):
                    for j in range(U_r.shape[1]):
                        val = float(U_r[i, j])
                        text_color = "white" if abs(val) > threshold else "black"
                        ax.text(
                            j,
                            i,
                            f"{val:.{precision}f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=8,
                        )

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Saved annotated U-matrix heatmaps to: {heatmap_path}")


# ============================================================
# Training
# ============================================================

def fit_choice_model(
    *,
    candidates: Sequence[int],
    rankings: List[Sequence[int]],
    counts: Sequence[int],
    lr: float = 0.05,
    steps: int = 100,
    device: str = "cuda",
    init_scale: float = 0.01,
    seed: Optional[int] = 0,
    log_every: int = 1,
    plot_loss: bool = False,
    rank_heterogeneous: bool = False,
    context_effects: bool = True,
    l2_lambda: float = 0.0,
    laplacian_lambda: float = 0.0,
    skip_zeros: bool = False
) -> Tuple[InteractionChoiceModel, List[float]]:

    if seed is not None:
        torch.manual_seed(seed)

    space = CandidateSpace(tuple(candidates))
    model = InteractionChoiceModel(
        space,
        init_scale=init_scale,
        rank_heterogeneous=rank_heterogeneous,
        context_effects=context_effects,
        l2_lambda=l2_lambda,
        laplacian_lambda=laplacian_lambda,
        skip_zeros=skip_zeros
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses, iters = [], []

    for t in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = model.nll_from_counts(rankings, counts)
        loss.backward()
        opt.step()

        if t % log_every == 0 or t == steps:
            losses.append(float(loss.detach().cpu()))
            iters.append(t)

    if plot_loss:
        fig, ax = plt.subplots()
        ax.plot(iters, losses)
        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("Weighted NLL (+ regularization)")
        ax.set_title("Training loss")
        fig.savefig("plots/choice-model-loss.pdf", bbox_inches="tight")

    return model, losses


if __name__ == "__main__":

    # burlington_filename = "data/preflib/elections-all/burlington/ED-00005-00000002.toi"
    # # burlington_filename = "data/preflib/elections-all/sf/ED-00021-00000007.toi"
    
    # ballots, ballot_counts, cand_names, skipped_votes = \
    #     utils.read_preflib(burlington_filename)


    # elim_votes = run_irv(len(cand_names), ballots.copy(), ballot_counts, cands=cand_names)
    # filtered_cands = utils.get_elim_order(elim_votes)[-4:]
    # filtered_cand_names = {cand: full_name.split(" ")[-1] for cand, full_name in cand_names.items() if cand in filtered_cands}
    # filtered_ballots, filtered_ballot_counts = utils.reduce_election(ballots, ballot_counts, filtered_cands)

    # visualize_ballot_tree(
    #     filtered_cand_names,
    #     filtered_ballots,
    #     filtered_ballot_counts,
    #     title="burlington-original"
    #     )

    # model, losses = fit_choice_model(
    #     candidates=list(filtered_cand_names.keys()),
    #     rankings=filtered_ballots,
    #     counts=filtered_ballot_counts,
    #     lr=0.1,
    #     steps=100,
    #     log_every=1,
    #     plot_loss=True,
    #     rank_heterogeneous=False
    # )

    # simulated_ballots = []
    # utils.build_tree([], filtered_cands, simulated_ballots)

    # n = np.sum(filtered_ballot_counts)
    # simulated_counts = []
    # for ballot in simulated_ballots:
    #     p_ballot = model.prob_ranking(ballot)
    #     simulated_counts.append(int(p_ballot * n))

    # visualize_ballot_tree(
    #     filtered_cand_names,
    #     simulated_ballots,
    #     tuple(simulated_counts),
    #     title="burlington-choice-model",
    #     aggregate_prefixes=True
    #     )

    # model.display_parameters(precision=3)


    # CHOICE MODEL EVAL WITH KL DIVERGENCE

    # burlington_filename = "data/preflib/elections-all/burlington/ED-00005-00000002.toi"
    # # burlington_filename = "data/preflib/elections-all/sf/ED-00021-00000007.toi"
    
    # ballots, ballot_counts, cand_names, skipped_votes = \
    #     utils.read_preflib(burlington_filename)
    # n = np.sum(ballot_counts)

    # cands = list(cand_names.keys())
    # all_possible_ballots = []
    # utils.build_tree([], cands, all_possible_ballots)

    # true_distribution = {
    #     tuple(ballot): ballot_counts[ballot_idx] / n
    #     for ballot_idx, ballot in enumerate(ballots)
    # }
    # for ballot in all_possible_ballots:
    #     if tuple(ballot) not in true_distribution:
    #         true_distribution[tuple(ballot)] = 0


    # sampling_rates = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]


    # bootstrap_div_means = []
    # rank_homo_div_means = []
    # rank_hetero_div_means = []

    # n_trials = 1

    # for sampling_rate in sampling_rates:
    #     sample_size = int(sampling_rate * n)
    #     bootstrap_div = []
    #     rank_homo_div = []
    #     rank_hetero_div = []

    #     for trial_num in range(n_trials):
    #         sample_counts = utils.resample(ballot_counts, sample_size, seed=trial_num)
            
    #         ## Bootstrap
    #         bootstrap_distribution = {
    #             tuple(ballot): sample_counts[ballot_idx]
    #             for ballot_idx, ballot in enumerate(ballots)
    #         }
    #         inferred_distribution = {}
    #         for ballot in all_possible_ballots:
    #             ballot_count = 1 # smoothing parameter
    #             if tuple(ballot) in bootstrap_distribution:
    #                 ballot_count += bootstrap_distribution[tuple(ballot)]

    #             inferred_distribution[tuple(ballot)] = ballot_count / (sample_size + len(all_possible_ballots))
    #         assert abs(np.sum(list(inferred_distribution.values())) - 1.0) < 1e-7

    #         bootstrap_div.append(utils.KL(true_distribution, inferred_distribution, all_possible_ballots))

    #         ## Rank Homo
    #         model, losses = fit_choice_model(
    #             candidates=list(cand_names.keys()),
    #             rankings=ballots,
    #             counts=sample_counts,
    #             lr=0.1,
    #             steps=50,
    #             log_every=1,
    #             plot_loss=False,
    #             rank_heterogeneous=False,
    #         )

    #         inferred_distribution = {}
    #         p_empty_ballot = model.prob_ranking([])
    #         for ballot in all_possible_ballots:
    #             p_ballot = model.prob_ranking(ballot)
    #             inferred_distribution[tuple(ballot)] = p_ballot / (1 - p_empty_ballot)
    #         rank_homo_div.append(utils.KL(true_distribution, inferred_distribution, all_possible_ballots))
    #         # rank_homo_div.append(0)

    #         ## Rank Hetero
    #         model, losses = fit_choice_model(
    #             candidates=list(cand_names.keys()),
    #             rankings=ballots,
    #             counts=sample_counts,
    #             lr=0.1,
    #             steps=50,
    #             log_every=1,
    #             plot_loss=False,
    #             rank_heterogeneous=True,
    #         )

    #         inferred_distribution = {}
    #         for ballot in all_possible_ballots:
    #             p_ballot = model.prob_ranking(ballot)
    #             inferred_distribution[tuple(ballot)] = p_ballot
    #         rank_hetero_div.append(utils.KL(true_distribution, inferred_distribution, all_possible_ballots))
    #         # rank_hetero_div.append(0)

    #     bootstrap_div_means.append(np.mean(bootstrap_div))
    #     rank_homo_div_means.append(np.mean(rank_homo_div))
    #     rank_hetero_div_means.append(np.mean(rank_hetero_div))

    # fig, ax = plt.subplots()

    # ax.plot(sampling_rates, bootstrap_div_means, label="Bootstrap")
    # ax.plot(sampling_rates, rank_homo_div_means, label="Rank Homogenous")
    # ax.plot(sampling_rates, rank_hetero_div_means, label="Rank Heterogeneous")

    # ax.set_xlabel("Sampling Rate")
    # ax.set_ylabel("KL Divergence with True Distribution")
    # ax.set_yscale("log")
    # ax.legend()
    # ax.grid()

    # fig.savefig("plots/compare_choice_models.pdf", bbox_inches="tight")

    # EVALUATE EFFICIENCY GAINS OF REMOVING BALLOTS WITH ZERO COUNT

    filename = "data/preflib/elections-all/burlington/ED-00005-00000002.toi"

    # burlington_filename = "data/preflib/elections-all/sf/ED-00021-00000007.toi"
    
    ballots, ballot_counts, cand_names, skipped_votes = \
        utils.read_preflib(filename)
    n = np.sum(ballot_counts)


    sample_size = int(n)
    sample_counts = utils.resample(ballot_counts, sample_size, with_replacement=False, seed=0)

    non_zero_ballots, non_zero_ballot_counts = utils.filter_zero_ballots(ballots, sample_counts)

    # start = time.time()
    # model, losses = fit_choice_model(
    #     candidates=list(cand_names.keys()),
    #     rankings=ballots,
    #     counts=sample_counts,
    #     lr=0.1,
    #     steps=50,
    #     log_every=1,
    #     plot_loss=False,
    #     rank_heterogeneous=True,
    # )
    # print(f"Runtime w/o filtering: {round(time.time() - start, 3)}")

    start = time.time()
    model, losses = fit_choice_model(
        candidates=list(cand_names.keys()),
        rankings=non_zero_ballots,
        counts=non_zero_ballot_counts,
        lr=0.1,
        steps=50,
        log_every=1,
        plot_loss=True,
        rank_heterogeneous=True,
        skip_zeros=True
    )
    print(f"Runtime w/ filtering: {round(time.time() - start, 3)}")