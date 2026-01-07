import sys
import numpy as np
import pickle
from tqdm import tqdm

import utils 

from push_pull_choice_model import fit_choice_model
from irv import run_irv

seed = 0
k = 5
size = 100

max_cands = 6

training_steps = 50

l2_lams = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
laplacian_lams = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

def get_cross_val_chunks(ballot_counts, k, size=200):
	'''
		Samples without replacement k ballot chunks, each with 'size' ballots.
		Returns a list of k ballot_counts.
	'''
	ballot_counts = np.array(ballot_counts)
	chunks = []
	for _ in range(k):
		chunk = np.array(
			utils.resample(ballot_counts, size, with_replacement=False, seed=seed))

		ballot_counts -= chunk
		chunks.append(chunk)

	assert np.sum([np.sum(chunk) for chunk in chunks]) == k * size
	return chunks

def get_cross_val_nll(ballots, count_chunks, val_chunk_idx,
	cands,
	l2_lambda, laplacian_lambda):
	'''
		train a push pull model where the training counts are the sum of all count_chunks except for val_chunk_idx
		return the nll of the validation counts on the trained model.
	'''
	
	training_counts = np.zeros(len(ballots))
	for idx, chunk in enumerate(count_chunks):
		if idx == val_chunk_idx:
			continue
		training_counts += chunk
	training_counts = tuple(training_counts)
	val_counts = tuple(count_chunks[val_chunk_idx])

	filtered_cands = cands.copy()
	filtered_training_ballots = ballots.copy()
	filtered_val_ballots = ballots.copy()

	num_cands = len(cands)

	if num_cands > max_cands:
		elim_votes = run_irv(num_cands, ballots.copy(), training_counts, cands=cands)
		filtered_cands = utils.get_elim_order(elim_votes)[-max_cands:]
		filtered_training_ballots, training_counts = utils.reduce_election(ballots.copy(), training_counts, filtered_cands)
		filtered_val_ballots, val_counts = utils.reduce_election(ballots.copy(), val_counts, filtered_cands)

	non_zero_ballots, non_zero_ballot_counts = utils.filter_zero_ballots(
		filtered_training_ballots,
		training_counts)

	model, _ = fit_choice_model(
		candidates=filtered_cands,
		rankings=non_zero_ballots,
		counts=non_zero_ballot_counts,
		lr=0.1,
		steps=training_steps,
		log_every=1e10, # don't log
		rank_heterogeneous=True,
		context_effects=True,
		l2_lambda=l2_lambda,
		laplacian_lambda=laplacian_lambda
		)

	nll = 0.0
	for idx, ballot in enumerate(filtered_val_ballots):
		if val_counts[idx] == 0:
			continue

		nll += val_counts[idx] * model.log_prob_ranking(ballot)
	return -nll


if __name__ == "__main__":

	assert len(sys.argv) == 2
	target_election = sys.argv[1]

	elections = utils.load_all_preflib_elections(election_dir=f"data/preflib/elections-all")
	
	target_idx = -1
	for idx, election_tup in enumerate(elections):
		election_name = f"{election_tup[0]}-{election_tup[1][-6:-4]}"
		if election_name == target_election:
			target_idx = idx
			break

	assert target_idx != -1
	ballots, ballot_counts, cand_names = elections[target_idx][2:5]
	cands = list(cand_names.keys())

	count_chunks = get_cross_val_chunks(ballot_counts, k=k, size=size)
	assert len(count_chunks) == k

	nlls = {}
	for l2_lam in tqdm(l2_lams):
		nlls[l2_lam] = {}
		for laplacian_lam in tqdm(laplacian_lams):
			nlls[l2_lam][laplacian_lam] = [
				get_cross_val_nll(ballots, count_chunks, chunk_idx, cands, l2_lam, laplacian_lam) \
				for chunk_idx in range(k)
			]

	with open(f"results/hyperparam/push_pull/{target_election}.pickle", "wb") as pickleFile:
		pickle.dump(nlls, pickleFile)





