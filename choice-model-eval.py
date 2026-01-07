import time 
from math import factorial
import pickle
from tqdm import tqdm

import numpy as np

import sys 
import os 

from collections import Counter

from irv import run_irv

import utils 

from push_pull_choice_model import fit_choice_model

# elim_votes = run_irv(k,
# 	ballots.copy(),
# 	ballot_counts,
# 	cands=cand_names)

########### Implementation notes ###########
# Max k = 6, SF-6 election
# IRV runs in <1 s for all elections
# Truncate to 9 candidates so that the size the state space is < 10^6
# For forecasting, the model inference is still the most time consuming step. e.g. simulation is ~0.01s for sf-06

########### MACROS ###########
assert len(sys.argv) == 4
target_election = sys.argv[1]
optimized_l2_lambda = float(sys.argv[2])
optimized_laplacian_lambda = float(sys.argv[3])

data_dir = "elections-all"
NUM_TRIALS = 75
# model_names = ["Bootstrap", "PL", "PL + Rank", "PL + Context", "PL + Rank + Context", "PL + Rank + Context + Reg"]
model_names = ["Bootstrap", "PL + Context",  "PL + Rank + Context + Reg"]


# model_names = ["PL + Rank + Context + Reg"]

max_k = 6
training_steps = 30
num_forecasting_simulations = 250

# sampling_rates = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.95, 1.0]
sample_sizes = np.array([50, 100, 250, 500, 1000, 2000, 4000])
# sample_sizes = np.array([100])


########### HELPERS ###########

def get_inferred_distribution(model_name, cands, ballots, ballot_counts, ballot_support):
	'''
	Train the model named model_name with ballots and ballot_counts as input data.
	Return a dictionary mapping each ballot in ballot_support to a probability
	'''

	if model_name == "Bootstrap":
		bootstrap_distribution = {
			tuple(ballot): ballot_counts[ballot_idx]
			for ballot_idx, ballot in enumerate(ballots)
		}

		inferred_distribution = {}
		num_possible_ballots = _get_num_possible_ballots(len(cands))
		sample_size = np.sum(ballot_counts)
		assert sample_size > 0

		for ballot in ballot_support:
			ballot_count = 0
			if tuple(ballot) in bootstrap_distribution:
				ballot_count = bootstrap_distribution[tuple(ballot)]

			inferred_distribution[tuple(ballot)] = ballot_count / sample_size
		return inferred_distribution
	else:
		rank_heterogeneous = "Rank" in model_name
		context_effects = "Context" in model_name 
		l2_lambda = optimized_l2_lambda if "Reg" in model_name else 0.0
		laplacian_lambda = optimized_laplacian_lambda if "Reg" in model_name else 0.0
		model, losses = fit_choice_model(
			candidates=cands,
			rankings=ballots,
			counts=ballot_counts,
			lr=0.1,
			steps=training_steps,
			log_every=1e10, # don't log
			rank_heterogeneous=rank_heterogeneous,
			context_effects=context_effects,
			l2_lambda=l2_lambda,
			laplacian_lambda=laplacian_lambda
		)

		inferred_distribution = {}
		p_empty_ballot = model.prob_ranking([])
		for ballot in ballot_support:
			p_ballot = model.prob_ranking(ballot)
			inferred_distribution[tuple(ballot)] = p_ballot / (1 - p_empty_ballot)
		return inferred_distribution

def _get_num_possible_ballots(k):
	total = 0
	for h in range(1, k + 1):
		total += factorial(k) / factorial(k - h)
	return total 

def _write_to_file(obj, filename):
	with open(filename, "wb") as pickleFile:
		pickle.dump(obj, pickleFile)

if __name__ == "__main__":

	elections = utils.load_all_preflib_elections(election_dir=f"data/preflib/{data_dir}")

	KL_by_election = {}
	inferred_distribution_by_election = {}

	candidate_win_shares_by_election = {}
	os.makedirs("results/push_pull_eval", exist_ok=True)

	for collection, file_name, ballots, ballot_counts, cand_names, _ in elections:
		election_name = f"{collection}-{file_name[-6:-4]}"
		if target_election != "" and election_name != target_election:
			continue
		print(f"Processing: {election_name}")
		
		# Election pre-processing
		n = np.sum(ballot_counts)
		k = len(cand_names)
		cands = list(cand_names.keys())
		assert np.max(sample_sizes) <= n

		true_distribution = {
			tuple(ballot): ballot_counts[ballot_idx] / n
			for ballot_idx, ballot in enumerate(ballots)
		}

		# Setup results data structures 
		KL_by_sample_size = {
			model_name: {sample_size: [] for sample_size in sample_sizes}
			for model_name in model_names
		}

		inferred_distribution_by_sample_size = {
			model_name: {sample_size: [] for sample_size in sample_sizes}
			for model_name in model_names
		}

		cand_wins_share_by_sample_size = {
			model_name: {
					sample_size: {
						cand: np.zeros(NUM_TRIALS) for cand in cands # crucially, there is no filtering at this stage
					} for sample_size in sample_sizes
				}
			for model_name in model_names
		}

		for sample_size in tqdm(sample_sizes):
			num_trials = NUM_TRIALS
			if sample_size >= 1000:
				num_trials = 10		
			for trial_num in tqdm(range(num_trials)):
				sample_counts = utils.resample(ballot_counts, sample_size, with_replacement=False, seed=trial_num)

				filtered_cands = cands.copy()
				filtered_cand_names = cand_names.copy()
				filtered_ballots = ballots.copy()
				filtered_counts = sample_counts
				if k > max_k:
					elim_votes = run_irv(k, ballots.copy(), sample_counts, cands=cand_names)
					filtered_cands = utils.get_elim_order(elim_votes)[-max_k:]
					filtered_cand_names = {cand: full_name for cand, full_name in cand_names.items() if cand in filtered_cands}
					filtered_ballots, filtered_counts = utils.reduce_election(ballots, sample_counts, filtered_cands)
				non_zero_ballots, non_zero_ballot_counts = utils.filter_zero_ballots(
					filtered_ballots.copy(),
					filtered_counts)

				all_possible_ballots = []
				utils.build_tree([], filtered_cands, all_possible_ballots)

				for model_name in model_names:
					
					# Distribution Difference 
					inferred_distribution = get_inferred_distribution(
						model_name,
						filtered_cands,
						non_zero_ballots.copy(), #only for training 
						non_zero_ballot_counts, #only for training
						ballot_support=all_possible_ballots.copy())
					KL_by_sample_size[model_name][sample_size].append(
						utils.l2(true_distribution, inferred_distribution)
					)
					inferred_distribution_by_sample_size[model_name][sample_size].append(inferred_distribution)


					## Forecasting 
					for forecasting_trial in range(num_forecasting_simulations):
						simulated_ballots, simulated_counts = utils.get_ballot_sample_from_distribution(
							inferred_distribution,
							num_samples=n,
							seed=forecasting_trial)

						elim_votes = run_irv(
							len(filtered_cands), 
							simulated_ballots.copy(), 
							simulated_counts, 
							cands=filtered_cands)
						winner = max(elim_votes, key=elim_votes.get)
						cand_wins_share_by_sample_size[model_name][sample_size][winner][trial_num] += 1

			filename = f"results/push_pull_eval/win_shares_by_election_200_trials_{target_election}.pickle"
			candidate_win_shares_by_election[election_name] = cand_wins_share_by_sample_size
			_write_to_file(candidate_win_shares_by_election, filename)

			filename = f"results/push_pull_eval/l2_by_election_200_trials_{target_election}.pickle"
			KL_by_election[election_name] = KL_by_sample_size
			_write_to_file(KL_by_election, filename)

			filename = f"results/push_pull_eval/inferred_distribution_by_election_200_trials_{target_election}.pickle"
			inferred_distribution_by_election[election_name] = inferred_distribution_by_sample_size
			_write_to_file(inferred_distribution_by_election, filename)