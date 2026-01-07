import time 
from math import factorial
import pickle
from tqdm import tqdm

import numpy as np

from collections import Counter

from irv import run_irv

import utils 

from push_pull_choice_model import fit_choice_model

'''
For each real-world election, according to the oracle election profile, how often does each candidate win for a given sampling rate?
'''

########### MACROS ###########
target_elections = ["burlington-02", "glasgow-04", "sf-11", "pierce-03"]
data_dir = "elections-all"
num_trials = 1000
model_names = ["Bootstrap"]
# model_names = ["Bootstrap", "PL + Rank + Context"]

sampling_rates = [1.0]

########### HELPERS ###########

def _write_win_shares_to_file(win_shares):
	existing_win_shares = {}
	try:
		with open(f"results/push_pull_eval/win_shares_by_election_oracle.pickle", "rb") as pickleFile:
			existing_win_shares = pickle.load(pickleFile)
	except:
		pass

	for election_name in win_shares:
		existing_win_shares[election_name] = win_shares[election_name]

	with open(f"results/push_pull_eval/win_shares_by_election_oracle.pickle", "wb") as pickleFile:
		pickle.dump(existing_win_shares, pickleFile)

if __name__ == "__main__":

	elections = utils.load_all_preflib_elections(election_dir=f"data/preflib/{data_dir}")

	candidate_win_shares_by_election = {}

	for collection, file_name, ballots, ballot_counts, cand_names, _ in elections:
		election_name = f"{collection}-{file_name[-6:-4]}"
		if election_name not in target_elections:
			continue
		print(f"Processing: {election_name}")
		
		# Election pre-processing
		n = np.sum(ballot_counts)
		k = len(cand_names)
		cands = list(cand_names.keys())

		cand_wins_share_by_sampling_rate = {
			sampling_rate: {
				cand: 0 for cand in cands
			} for sampling_rate in sampling_rates
		}

		for sampling_rate in sampling_rates:
			sample_size = int(sampling_rate * n)
			
			for trial_num in tqdm(range(num_trials)):
				sample_counts = utils.resample(ballot_counts, sample_size, with_replacement=True, seed=trial_num)

				elim_votes = run_irv(
					k, 
					ballots.copy(), 
					sample_counts, 
					cands=cands)
				winner = max(elim_votes, key=elim_votes.get)
				cand_wins_share_by_sampling_rate[sampling_rate][winner] += (1 / num_trials)

			candidate_win_shares_by_election[election_name] = cand_wins_share_by_sampling_rate
			_write_win_shares_to_file(candidate_win_shares_by_election)

print(candidate_win_shares_by_election)








