import numpy as np
import os
import glob
from numpy.random import default_rng
from collections import defaultdict

import time
from tqdm import tqdm

colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']

def get_elim_order(elim_votes):
    '''
        Return the elimination order of an election as a string "cand_A cand_B ... cand_M" where cand_M is eliminated last (the winner).
        The input elim_votes is a dictionary mapping candidates to the number of votes they had at elimination.
    '''

    cands = np.array([str(cand) for cand in elim_votes.keys()])
    num_elim_votes = np.array(list(elim_votes.values()))

    sorted_idxs = np.argsort(num_elim_votes)
    return " ".join(cands[sorted_idxs])

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