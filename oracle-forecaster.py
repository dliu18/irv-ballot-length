import time 
from math import factorial
import pickle
from tqdm import tqdm

import numpy as np

from collections import Counter

from irv import run_irv_with_mov

import utils 

from push_pull_choice_model import fit_choice_model

'''
For each real-world election, according to the oracle election profile, how often does each candidate win for a given sampling rate?
'''

########### MACROS ###########
target_elections = ["burlington-02", "glasgow-04", "sf-11", "pierce-03"]
data_dir = "elections-all"
num_trials = 1000
stat_names = ["elimination rank", "elimination votes", "last round mov"]

########### HELPERS ###########


if __name__ == "__main__":

	elections = utils.load_all_preflib_elections(election_dir=f"data/preflib/{data_dir}")


	for collection, file_name, ballots, ballot_counts, cand_names, _ in elections:
		election_name = f"{collection}-{file_name[-6:-4]}"
		if election_name not in target_elections:
			continue
		print(f"Processing: {election_name}")
		
		# Election pre-processing
		n = np.sum(ballot_counts)
		k = len(cand_names)
		cands = list(cand_names.keys())

		election_stats = {
			stat_name: []
			for stat_name in stat_names
		}

		for trial_num in tqdm(range(num_trials)):
			sample_counts = utils.resample(ballot_counts, n, with_replacement=True, seed=trial_num)

			elim_votes, mov = run_irv_with_mov(
				k, 
				ballots.copy(), 
				sample_counts, 
				cands=cands)

			sorted_cands = sorted(elim_votes.items(), key=lambda x: x[1], reverse=True)
			elim_rank = {cand: rank + 1 for rank, (cand, _) in enumerate(sorted_cands)}

			election_stats["elimination rank"].append(elim_rank)
			election_stats["elimination votes"].append(elim_votes)
			election_stats["last round mov"].append(mov)

		filename = f"results/push_pull_eval/oracle_forecast_{election_name}.pickle"
		with open(filename, "wb") as pickleFile:
			pickle.dump(election_stats, pickleFile)





