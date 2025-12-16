import numpy as np
import os
import glob
from numpy.random import default_rng
from collections import defaultdict, Counter

import model 

import time
from tqdm import tqdm

colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']


def get_elim_order(elim_votes):
    '''
        Return the elimination order of an election as a string "cand_A cand_B ... cand_M" where cand_M is eliminated last (the winner).
        The input elim_votes is a dictionary mapping candidates to the number of votes they had at elimination.
    '''

    cands = np.array([cand for cand in elim_votes.keys()])
    num_elim_votes = np.array(list(elim_votes.values()))

    sorted_idxs = np.argsort(num_elim_votes)
    return cands[sorted_idxs]

def clean_up_invalid_ballots(ballots, ballot_counts):
    """
    Fix ballots where candidates appear multiple times. Only the first appearance of a candidate is counted
    :param ballots:
    :param ballot_counts:
    :return:
    """

    merged_counts = defaultdict(int)

    for ballot, ballot_count in zip(ballots, ballot_counts):
        clean_ballot, idx = np.unique(ballot, return_index=True)
        clean_ballot = tuple(clean_ballot[np.argsort(idx)])

        merged_counts[clean_ballot] += ballot_count

    ballots, ballot_counts = zip(*merged_counts.items())

    return list(map(np.array, ballots)), ballot_counts

def reduce_election(ballots, ballot_counts, filtered_cands):
    '''
        Reduce an election to the candidates in filtered_cands. Return the reduced ballots and ballot_counts.
    '''

    ctr = Counter()
    for ballot_idx, ballot in enumerate(ballots):
        filtered_ballot = [cand for cand in ballot if cand in filtered_cands]
        if len(filtered_ballot) > 0:
            ctr[tuple(filtered_ballot)] += ballot_counts[ballot_idx]

    items = ctr.most_common()
    output_ballots = [np.array(list(tpl)) for tpl, _ in items]
    ballot_counts = tuple([cnt for _, cnt in items])

    return output_ballots, ballot_counts


def resample(ballot_counts, sample_size=-1, with_replacement=True, seed=0):
    rng = default_rng(seed=seed)
    n = np.sum(ballot_counts)
    if sample_size == -1:
        sample_size = n

    if with_replacement:
        p = np.array(ballot_counts) / n
        return rng.multinomial(sample_size, pvals=p)
    else:
        return rng.multivariate_hypergeometric(ballot_counts, sample_size)
        # assert sample_size <= n

        # indices = np.concatenate([
        #     [idx] * ballot_counts[idx] 
        #     for idx in range(len(ballot_counts))
        # ])

        # sampled_idxs = rng.choice(
        #     indices,
        #     size=sample_size,
        #     replace=False)

        # resampled_counts = np.zeros(len(ballot_counts))
        # for sampled_idx in sampled_idxs:
        #     resampled_counts[sampled_idx] += 1

        # assert np.sum(resampled_counts) == sample_size
        # return resampled_counts


def load_all_preflib_elections(election_dir=""):
    elections = []

    for collection in glob.glob(f'{election_dir}/*'):
        for file_name in glob.glob(f'{collection}/*.toi') + glob.glob(f'{collection}/*.soi'):

            # Skip duplicate elections with write-ins
            if os.path.basename(file_name) in ['ED-00018-00000001.soi', 'ED-00018-00000003.soi']:
                continue

            ballots, ballot_counts, cand_names, skipped_votes = read_preflib(file_name)

            elections.append((
                os.path.basename(collection),
                os.path.basename(file_name),
                ballots,
                ballot_counts,
                cand_names,
                skipped_votes
            ))

    return elections

def read_preflib(file_name):
    with open(file_name, 'r') as f:
        n_cands = int(f.readline())

        cand_names = dict()
        for i in range(n_cands):
            split = f.readline().strip().split(',')
            cand_idx = int(split[0])
            cand_names[cand_idx] = ','.join(split[1:])

        n_voters, votes, unique_ballots = map(int, f.readline().strip().split(','))

        ballot_counts = []
        ballots = []
        skipped_votes = 0
        for i in range(unique_ballots):
            line = f.readline().strip()
            split_line = line.split(',')

            # Skip any ballots with ties
            if '{' not in line:
                ballot_counts.append(int(split_line[0]))
                ballots.append(np.array(tuple(map(int, split_line[1:]))))
            else:
                skipped_votes += int(split_line[0])

    ballots, ballot_counts = clean_up_invalid_ballots(ballots, ballot_counts)

    return ballots, ballot_counts, cand_names, skipped_votes


def get_model_object(model_name, cands):
    if model_name == "Bootstrap":
        return model.Bootstrap(cands)
    #
    elif model_name == "PL":
        return model.PLModel(cands, use_end_marker=True, include_unranked=True)
    #
    elif model_name == "Contextual":
        return model.ContextModel(cands)
    elif model_name == "Contextual By Length":
        return model.ContextModel(cands, first_choice_prob_by_length=True)
    elif model_name == "Contextual By Length and Smoothing 1":
        return model.ContextModel(cands, first_choice_prob_by_length=True, smoothing=1)
    elif model_name == "Contextual By Length and Smoothing 5":
        return model.ContextModel(cands, first_choice_prob_by_length=True, smoothing=5)
    elif model_name == "Contextual Truncated":
        return model.ContextModel(cands, truncation_h=4)
    elif model_name == "Contextual Truncated Weighted First":
        return model.ContextModel(cands, truncation_h=4, rank_weighted_first_choice=True)
    elif model_name == "Contextual Truncated Weighted Mtx":
        return model.ContextModel(cands, truncation_h=4, rank_weighted_transition_mtx=True)
    elif model_name == "Contextual Truncated Weighted First and Mtx":
        return model.ContextModel(cands, truncation_h=4, 
            rank_weighted_first_choice=True,
            rank_weighted_transition_mtx=True)
    #
    elif model_name == "Mallows Dispersion 0.5":
        return model.MallowsModel(cands, dispersion=0.5)
    elif model_name == "Mallows Dispersion 1":
        return model.MallowsModel(cands, dispersion=1)
    elif model_name == "Mallows Dispersion 5":
        return model.MallowsModel(cands, dispersion=5)
    #
    elif model_name == "Contextual Perturbation Dispersion 0.5":
        return model.ContextualPerturbationModel(cands, dispersion=0.5)
    elif model_name == "Contextual Perturbation Dispersion 1":
        return model.ContextualPerturbationModel(cands, dispersion=1)
    elif model_name == "Contextual Perturbation Dispersion 5":
        return model.ContextualPerturbationModel(cands, dispersion=5)
    #
    elif model_name == "Uniform":
        return model.UniformModel(cands)
    else:
        raise ValueError(f"Unknown model_name '{model_name}'")

# ---------------------------------------------------------------------
# Distance functions for partial rankings
# ---------------------------------------------------------------------

def kendall_tau_distance_partial(ballot1, ballot2, all_cands):
    """
    Kendall tau distance for partial rankings.

    Only candidate pairs that are ranked in BOTH ballots are considered.
    Unranked / missing candidates are ignored.

    Parameters
    ----------
    ballot1, ballot2 : sequence
        Partial rankings (lists/arrays) of candidate ids.

    Returns
    -------
    int
        Number of pairwise inversions between ballot1 and ballot2.
    """
    pos1 = {cand: idx for idx, cand in enumerate(ballot1)}
    pos2 = {cand: idx for idx, cand in enumerate(ballot2)}
    common = [c for c in pos1 if c in pos2]

    dist = 0
    for i in range(len(common)):
        ci = common[i]
        for j in range(i + 1, len(common)):
            cj = common[j]
            i1, j1 = pos1[ci], pos1[cj]
            i2, j2 = pos2[ci], pos2[cj]
            # Orders disagree if the sign of (i1 - j1) and (i2 - j2) differ
            if (i1 - j1) * (i2 - j2) < 0:
                dist += 1
    return dist

def footrule_distance_partial(ballot1, ballot2, all_cands):
    """
    Spearman footrule distance for partial rankings.

    All candidates in self.all_cands are considered.
    Unranked candidates are assigned a default rank equal to len(self.all_cands),
    so default rank is consistent across ballots.

    Parameters
    ----------
    ballot1, ballot2 : sequence
        Partial rankings (lists/arrays) of candidate ids.

    Returns
    -------
    int
        Sum of absolute rank differences across candidates.
    """
    pos1 = {cand: idx for idx, cand in enumerate(ballot1)}
    pos2 = {cand: idx for idx, cand in enumerate(ballot2)}

    default_rank = len(all_cands)
    dist = 0
    for c in all_cands:
        r1 = pos1.get(c, default_rank)
        r2 = pos2.get(c, default_rank)
        dist += abs(r1 - r2)
    return dist

def edit_distance_partial(ballot1, ballot2, all_cands):
    """
    Permutation edit distance (Levenshtein) for ballot sequences.

    We treat ballots as sequences of candidate ids and compute the minimum
    number of insertions, deletions, or substitutions needed to transform
    ballot1 into ballot2.

    Parameters
    ----------
    ballot1, ballot2 : sequence
        Partial rankings (lists/arrays) of candidate ids.

    Returns
    -------
    int
        Edit distance between the two ballots.
    """
    m, n = len(ballot1), len(ballot2)
    if m == 0:
        return n
    if n == 0:
        return m

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ballot1[i - 1] == ballot2[j - 1]:
                cost_sub = 0
            else:
                cost_sub = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost_sub  # substitution
            )

    return dp[m][n]


if __name__ == "__main__":

    ## get_elim_order
    elim_votes = {
        "A": 10,
        "B": 5,
        "C": 30
    }

    elim_order = get_elim_order(elim_votes)
    assert elim_order == "B A C"


    ## Sampling runtime analysis

    file_name = "data/preflib/elections-all/burlington/ED-00005-00000002.toi" #burlington-election
    ballots, ballot_counts, cand_names, skippped_votes = read_preflib(file_name)
    n = np.sum(ballot_counts)

    # With Replacement
    for _ in tqdm(range(500)):
        resampled_ballots = resample(ballot_counts, sample_size=n, with_replacement=True)

    # Without Replacement
    for _ in tqdm(range(500)):
        resampled_ballots = resample(ballot_counts, sample_size=n, with_replacement=False)