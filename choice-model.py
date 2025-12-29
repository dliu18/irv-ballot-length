from __future__ import annotations

import numpy as np

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import utils 
from visualize_ballot_trees import visualize_ballot_tree 
from irv import run_irv

@dataclass(frozen=True)
class CandidateSpace:
    """
    Maps arbitrary integer candidate IDs to dense indices 0..m-1 for internal use.
    """
    candidates: Tuple[int, ...]

    def __post_init__(self):
        # enforce uniqueness while preserving order
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidates must be unique.")
        if len(self.candidates) == 0:
            raise ValueError("candidates cannot be empty.")

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
        """
        Convert candidate IDs -> internal indices. Also validates:
          - all candidates are in the candidate set
          - no repeats
        """
        id2i = self.id_to_idx
        out: List[int] = []
        seen = set()
        for cid in ranking:
            if cid not in id2i:
                raise ValueError(f"Ranking contains candidate {cid} not in candidates={self.candidates}.")
            if cid in seen:
                raise ValueError(f"Ranking contains repeated candidate {cid}: {ranking}")
            seen.add(cid)
            out.append(id2i[cid])
        return out


class FullPrefixChoiceModel(nn.Module):
    """
    Exact discrete choice model with free parameters per (prefix-state, action).

    State: bitmask over m candidates indicating which have been chosen.
    Actions: choose any remaining candidate, or END.

    logits[state, action] are learned. Invalid candidate actions are masked out.
    """
    def __init__(self, candidate_space: CandidateSpace, init_scale: float = 0.01):
        super().__init__()
        self.space = candidate_space
        self.m = candidate_space.m
        self.num_states = 1 << self.m
        self.end_action = self.m
        self.num_actions = self.m + 1  # last is END

        logits = init_scale * torch.randn(self.num_states, self.num_actions)
        self.logits = nn.Parameter(logits)

        # Precompute invalid masks: True => invalid action.
        invalid_masks = torch.zeros(self.num_states, self.num_actions, dtype=torch.bool)
        for s in range(self.num_states):
            for j in range(self.m):
                if (s >> j) & 1:
                    invalid_masks[s, j] = True
            invalid_masks[s, self.end_action] = False  # END always valid
        self.register_buffer("invalid_masks", invalid_masks)

    def _log_probs_at_state(self, state: int) -> torch.Tensor:
        """
        Return log-probs over actions for a given state with invalid candidates masked out.
        Shape: (m+1,)
        """
        masked = self.logits[state].masked_fill(self.invalid_masks[state], float("-inf"))
        return F.log_softmax(masked, dim=-1)

    def log_prob_ranking_idx(self, ranking_idx: Sequence[int]) -> torch.Tensor:
        """
        ranking_idx: list of internal indices (0..m-1), unique.
        Returns scalar tensor: log P(ranking then END).
        """
        device = self.logits.device
        state = 0
        logp = torch.tensor(0.0, device=device)

        for a in ranking_idx:
            a = int(a)
            if not (0 <= a < self.m):
                raise ValueError(f"Internal action out of range: {a}. Expected 0..{self.m-1}.")
            if (state >> a) & 1:
                raise ValueError(f"Internal ranking repeats candidate index {a}: {ranking_idx}")
            log_probs = self._log_probs_at_state(state)
            logp = logp + log_probs[a]
            state = state | (1 << a)

        # END
        log_probs = self._log_probs_at_state(state)
        logp = logp + log_probs[self.end_action]
        return logp

    def prob_ranking(self, ranking: Sequence[int]) -> float:
        """
        Public API: ranking is a list of candidate IDs (as provided in candidate_space).
        Returns P(ranking) as a Python float.
        """
        ranking_idx = self.space.encode_ranking(ranking)
        with torch.no_grad():
            lp = self.log_prob_ranking_idx(ranking_idx)
            return float(torch.exp(lp).cpu().item())

    def log_prob_ranking(self, ranking: Sequence[int]) -> float:
        """
        Public API: returns log P(ranking) as Python float.
        """
        ranking_idx = self.space.encode_ranking(ranking)
        with torch.no_grad():
            lp = self.log_prob_ranking_idx(ranking_idx)
            return float(lp.cpu().item())

    def nll_from_counts(self, rankings: List[Sequence[int]], counts: Sequence[int]) -> torch.Tensor:
        """
        Weighted negative log-likelihood normalized by total counts.
        Minimizing this is equivalent to minimizing KL(P_data || P_model).
        """
        if len(rankings) != len(counts):
            raise ValueError("rankings and counts must have the same length.")
        if len(rankings) == 0:
            raise ValueError("rankings cannot be empty.")

        device = self.logits.device
        counts_t = torch.tensor(counts, dtype=torch.float32, device=device)
        if (counts_t < 0).any():
            raise ValueError("counts must be nonnegative.")
        total = counts_t.sum().clamp_min(1.0)

        logps = []
        for idx, r in enumerate(rankings):
            r_idx = self.space.encode_ranking(r)
            logps.append(self.log_prob_ranking_idx(r_idx))
        logps_t = torch.stack(logps)  # (N,)

        return -(counts_t * logps_t).sum() / total


def fit_choice_model(
    *,
    candidates: Sequence[int],
    rankings: List[Sequence[int]],
    counts: Sequence[int],
    lr: float = 0.05,
    steps: int = 2000,
    device: str = "cuda",
    init_scale: float = 0.01,
    seed: Optional[int] = 0,
    log_every: int = 100,
    plot_loss: bool = False,
) -> Tuple[FullPrefixChoiceModel, List[float]]:
    """
    Fits the model by minimizing KL(P_data || P_model) (equiv. weighted NLL on observed rankings).

    Returns:
      model, losses (recorded every log_every steps, including final step if aligned)
    """
    if seed is not None:
        torch.manual_seed(seed)

    space = CandidateSpace(tuple(candidates))
    model = FullPrefixChoiceModel(space, init_scale=init_scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses: List[float] = []
    iters: List[int] = []

    for t in tqdm(range(1, steps + 1)):
        opt.zero_grad(set_to_none=True)
        loss = model.nll_from_counts(rankings, counts)
        loss.backward()
        opt.step()

        if (t % log_every) == 0 or t == steps:
            losses.append(float(loss.detach().cpu().item()))
            iters.append(t)

    if plot_loss:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(iters, losses)
        plt.xlabel("Optimization step")
        plt.ylabel("Weighted NLL (cross-entropy)")
        plt.title("Loss curve")
        plt.savefig("choice-model-loss.pdf", bbox_inches="tight")

    return model, losses


def build_tree(prefix, cands, nodes):
    if len(prefix) == len(cands):
        return

    unseen = [cand for cand in cands if cand not in prefix]
    for cand in unseen:
        next_prefix = prefix + [cand]
        nodes.append(next_prefix)
        build_tree(next_prefix, cands, nodes)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # candidates = [10, 20, 30, 40]  # arbitrary integer IDs

    # rankings = [
    #     [10, 20],
    #     [10],
    #     [30, 40, 20],
    # ]
    # counts = [10, 5, 3]

    burlington_filename = "data/preflib/elections-all/burlington/ED-00005-00000002.toi"
    # burlington_filename = "data/preflib/elections-all/sf/ED-00021-00000007.toi"
    
    ballots, ballot_counts, cand_names, skipped_votes = \
        utils.read_preflib(burlington_filename)


    elim_votes = run_irv(len(cand_names), ballots.copy(), ballot_counts, cands=cand_names)
    filtered_cands = utils.get_elim_order(elim_votes)[-4:]
    filtered_cand_names = {cand: full_name.split(" ")[-1] for cand, full_name in cand_names.items() if cand in filtered_cands}
    filtered_ballots, filtered_ballot_counts = utils.reduce_election(ballots, ballot_counts, filtered_cands)

    visualize_ballot_tree(
        filtered_cand_names,
        filtered_ballots,
        filtered_ballot_counts,
        title="burlington-original"
        )

    model, losses = fit_choice_model(
        candidates=list(filtered_cand_names.keys()),
        rankings=filtered_ballots,
        counts=filtered_ballot_counts,
        lr=0.1,
        steps=100,
        log_every=1,
        plot_loss=True,
    )

    simulated_ballots = []
    build_tree([], filtered_cands, simulated_ballots)

    n = np.sum(filtered_ballot_counts)
    simulated_counts = []
    for ballot in simulated_ballots:
        p_ballot = model.prob_ranking(ballot)
        simulated_counts.append(int(p_ballot * n))

    visualize_ballot_tree(
        filtered_cand_names,
        simulated_ballots,
        tuple(simulated_counts),
        title="burlington-choice-model"
        )
