import time 
from math import factorial
import pickle
from tqdm import tqdm

import numpy as np

import sys 
import os 
import multiprocessing

from collections import Counter

from irv import run_irv_with_mov, run_irv

import utils 

from push_pull_choice_model import fit_choice_model

########### MACROS ###########
assert len(sys.argv) == 3
target_election = sys.argv[1]
exp_name = sys.argv[2]

data_dir = "elections-all"
stat_names = ["elimination rank", "elimination votes", "last round mov"]

# model_names = ["PL + Rank + Context + Reg"]

max_k = 6
num_forecasting_simulations = 100

########### HELPERS ###########

def _get_num_possible_ballots(k):
	total = 0
	for h in range(1, k + 1):
		total += factorial(k) / factorial(k - h)
	return total 

def _write_to_file(obj, filename):
	with open(filename, "wb") as pickleFile:
		pickle.dump(obj, pickleFile)


def _process_sample_size(args):
	sample_size, inferred_distribution_by_model, model_names, ballots, ballot_counts, cand_names, cands, k, n, max_k, num_forecasting_simulations = args
	
	sample_stats_by_model = {
		model_name: {sample_size: {stat_name: [] for stat_name in stat_names}}
		for model_name in model_names
	}
	
	num_trials = len(inferred_distribution_by_model[model_names[0]][sample_size])
	
	for trial_num in range(num_trials):
		sample_counts = utils.resample(ballot_counts, sample_size, with_replacement=True, seed=trial_num)
		
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
			inferred_distribution = inferred_distribution_by_model[model_name][sample_size][trial_num]
			
			## Forecasting
			for forecasting_trial in range(num_forecasting_simulations):
				start = time.time() 
				simulated_ballots, simulated_counts = utils.get_ballot_sample_from_distribution(
					inferred_distribution,
					num_samples=n,
					seed=forecasting_trial)
				# print(f"Sampling time: {round(time.time() - start, 3)}")
				
				simulated_ballots, simulated_counts = utils.filter_zero_ballots(
					simulated_ballots.copy(),
					simulated_counts)

				start = time.time()
				elim_votes, mov = run_irv_with_mov(
					len(filtered_cands), 
					simulated_ballots.copy(), 
					simulated_counts, 
					cands=filtered_cands)
				# print(f"IRV time: {round(time.time() - start, 3)}")
				
				start = time.time()
				sorted_cands = sorted(elim_votes.items(), key=lambda x: x[1], reverse=True)
				elim_rank = {cand: rank + 1 for rank, (cand, _) in enumerate(sorted_cands)}
				elim_rank_all_cands = {cand: elim_rank[cand] if cand in elim_rank else max_k + 1 for cand in cands}
				
				elim_votes_all_cands = {cand: elim_votes[cand] if cand in elim_votes else 0 for cand in cands}
				
				sample_stats_by_model[model_name][sample_size]["elimination rank"].append(elim_rank_all_cands)
				sample_stats_by_model[model_name][sample_size]["elimination votes"].append(elim_votes_all_cands)
				sample_stats_by_model[model_name][sample_size]["last round mov"].append(mov)
				# print(f"Agg time: {round(time.time() - start, 3)}")
	
	return sample_size, sample_stats_by_model


def _validate_training_output(inferred_distribution_by_model):
	'''
	Verify:
		* all model names share the same list of sample sizes
		* for a given sample size, all models have the same number of trials

	Return the list of model names and sample sizes
	'''

	model_names = list(inferred_distribution_by_model.keys())
	sample_sizes = np.sort(list(inferred_distribution_by_model[model_names[0]].keys()))
	
	# all models have the same sample sizes
	for model_name in model_names[1:]:
		assert (sample_sizes == np.sort(list(inferred_distribution_by_model[model_name].keys()))).all()
	
	for sample_size in sample_sizes:
		num_trials = len(inferred_distribution_by_model[model_names[0]][sample_size])
		for model_name in model_names[1:]:
			assert num_trials == len(inferred_distribution_by_model[model_name][sample_size])
	return model_names, list(sample_sizes)

if __name__ == "__main__":

	elections = utils.load_all_preflib_elections(election_dir=f"data/preflib/{data_dir}")

	for collection, file_name, ballots, ballot_counts, cand_names, _ in elections:
		election_name = f"{collection}-{file_name[-6:-4]}"
		if target_election != "" and election_name != target_election:
			continue
		print(f"Processing: {election_name}")
		

		inferred_distribution_filename = f"results/push_pull_eval/inferred_distribution_by_election_{exp_name}_{target_election}.pickle"
		with open(inferred_distribution_filename, "rb") as pickleFile:
			inferred_distribution_by_model = pickle.load(pickleFile)[election_name]

		model_names, sample_sizes = _validate_training_output(inferred_distribution_by_model)

		# Election pre-processing
		n = np.sum(ballot_counts)
		k = len(cand_names)
		cands = list(cand_names.keys())

		# Setup results data structures 
		election_stats_by_model = {
			model_name: {
				sample_size: {
					stat_name: []
					for stat_name in stat_names} 
				for sample_size in sample_sizes}
			for model_name in model_names
		}

		# Parallelize over sample sizes (each sample size on a separate CPU)
		args_list = [
			(sample_size, inferred_distribution_by_model, model_names, ballots, ballot_counts, cand_names, cands, k, n, max_k, num_forecasting_simulations)
			for sample_size in sample_sizes
		]
		processes = min(len(sample_sizes), multiprocessing.cpu_count()) if len(sample_sizes) > 0 else 1
		with multiprocessing.Pool(processes=processes) as pool:
			for sample_size, sample_stats_by_model in tqdm(pool.imap(_process_sample_size, args_list), total=len(sample_sizes)):
				for model_name in model_names:
					for stat_name in stat_names:
						election_stats_by_model[model_name][sample_size][stat_name].extend(
							sample_stats_by_model[model_name][sample_size][stat_name]
						)
				filename = f"results/push_pull_eval/estimator_forecast_{exp_name}_{target_election}.pickle"
				_write_to_file(election_stats_by_model, filename)

