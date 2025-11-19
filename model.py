import utils 
import time

import choix
import numpy as np 

from collections import Counter

EXTRA_RATIO = 1.5

class BaseModel:
	def __init__(self, cand_names, seed=0):
		'''
			cand_names is a dictionary of candidate ids to full names
		'''
		self.is_fitted=False
		self.k = len(cand_names)
		self.cand_names=cand_names.copy()
		self.num_cands=len(cand_names)

		self.seed=seed
		self.rng = np.random.default_rng(self.seed)


	def fit(self, ballots, ballot_counts):
		'''
			ballots is a list of ballots where each ballot is a list of ranked candidates.
			ballot_counts specifies the number of people ascribing to a particular ballot
		'''
		self.is_fitted=True


	def simulate_ballots(self, num_ballots, seed=0):
		assert self.is_fitted

class Bootstrap(BaseModel):
	def __init__(self, cand_names, seed=0):
		super().__init__(cand_names, seed)

	def fit(self, ballots, ballot_counts):
		self.is_fitted = True 
		self.ballots = ballots.copy()
		self.ballot_counts = ballot_counts

	def simulate_ballots(self, num_ballots, seed=0):
		simulated_counts = utils.resample(self.ballot_counts,
						sample_size=num_ballots,
						with_replacement=True,
						seed=seed)
		return self.ballots.copy(), list(simulated_counts)

class PLModel(BaseModel):
	def __init__(self, cand_names, seed=0, use_end_marker=True, include_unranked=True):
		super().__init__(cand_names, seed)
		self.use_end_marker = use_end_marker
		self.include_unranked = include_unranked

		if self.use_end_marker:
			self.cand_names["END"] = "END"

		# Mapping
		self.cand_id_to_idx = {cid: i for i, cid in enumerate(self.cand_names)}
		self.cand_idx_to_id = {i: cid for cid, i in self.cand_id_to_idx.items()}

	def _convert_ballots_to_ranking_data(self, ballots, ballot_counts):
		ranking_data = []
		for ballot_idx, ballot_with_ids in enumerate(ballots):
			### convert to indices
			ballot_with_idxs = [
				self.cand_id_to_idx[cand_id] 
				for cand_id in ballot_with_ids
			]
			if self.use_end_marker:
				ballot_with_idxs.append(self.num_cands) # end

			### handle unlisted candidates
			if self.include_unranked:
				listed = set(ballot_with_idxs)  # O(n)
				unlisted_cands = [i for i in range(self.num_cands) if i not in listed]

			for _ in range(ballot_counts[ballot_idx]):
				full_ballot = ballot_with_idxs.copy()
				if self.include_unranked and unlisted_cands:
					self.rng.shuffle(unlisted_cands)
					full_ballot.extend(unlisted_cands)
					assert len(full_ballot) == self.num_cands + 1
				ranking_data.append(full_ballot)

		assert len(ranking_data) == np.sum(ballot_counts)
		return ranking_data
		 
	def _convert_choix_ballots_for_output(self, simulated_ballots, num_ballots):
		'''
			simulated_ballots contains extra ballots for buffer. return only the first num_ballots valid ones.
		'''
		ctr = Counter()
		valid_ballots = 0
		for ballot in simulated_ballots:
			if self.use_end_marker:
				# find END index, skip if at front
				end_idx_arr = np.where(ballot == self.cand_id_to_idx["END"])[0]
				if len(end_idx_arr) == 0:
					end_idx = len(ballot)
				else:
					end_idx = int(end_idx_arr[0])
					if end_idx ==  0:
						continue
			else:
				# no END marker, use full ballot
				end_idx = len(ballot)
			truncated = tuple(map(int, ballot[:end_idx])) #mapping to int ensure the tuple is hashable
			ctr[truncated] += 1
			valid_ballots += 1

			if valid_ballots == num_ballots:
				break

		assert valid_ballots == num_ballots

		# Sort by count desc
		items = ctr.most_common()
		output_ballots = [np.array([self.cand_idx_to_id[i] for i in tpl]) for tpl, _ in items]
		ballot_counts = [cnt for _, cnt in items]

		assert np.all([len(ballot) > 0 for ballot in output_ballots])

		return output_ballots, ballot_counts
		
	def fit(self, ballots, ballot_counts, alpha=1e-3):
		self.is_fitted=True

		start = time.time()
		ranking_data = self._convert_ballots_to_ranking_data(ballots, ballot_counts)

		PL_params = choix.lsr.ilsr_rankings(
			n_items=self.num_cands + 1 if self.use_end_marker else self.num_cands,
			data=ranking_data,
			alpha=alpha
		)
		self.params = PL_params
		# print(
		# 	f"PL Params: \n", 
		# 	"\n".join([
		# 		f"{self.cand_names[self.cand_idx_to_id[cand_idx]]}:\t{weight}"
		# 		for cand_idx, weight in enumerate(PL_params)
		# 		])
		# 	)

	def simulate_ballots(self, num_ballots, seed=0):
		assert self.is_fitted
		
		#generate_rankings does not have a seed parameter. 
		#generate extra ballots to account for the discarded ballots that start with the END character
		simulated_ballots_from_choix = choix.utils.generate_rankings(
			self.params, 
			n_rankings=int(EXTRA_RATIO*num_ballots), 
			size=self.num_cands + 1 if self.use_end_marker else self.num_cands)

		return self._convert_choix_ballots_for_output(simulated_ballots_from_choix, num_ballots)


class ContextModel(BaseModel):
	def __init__(self, cand_names, seed=0, first_choice_prob_by_length=False):
		super().__init__(cand_names, seed)

		# probability each cand is the first choice in a ballot
		# includes the option to learn a separate distribution for each ballot length, where the length row indexes the matrix
		# however, by default, all ballot lengths share the same distribution.
		self.first_choice_prob = np.zeros((self.k, self.k))
		self.first_choice_prob_by_length = first_choice_prob_by_length 

		self.length_prob = np.zeros(self.k + 1) # discrete prob dist over ballot lengths. P(length = 0) = 0
		self.transition_mtx = np.zeros((self.k, self.k))

		# Mapping
		self.cand_id_to_idx = {cid: i for i, cid in enumerate(self.cand_names)}
		self.cand_idx_to_id = {i: cid for cid, i in self.cand_id_to_idx.items()}

	def get_params(self):
		return self.first_choice_prob.copy(), self.length_prob.copy(), self.transition_mtx.copy()

	def fit(self, ballots, ballot_counts):
		# first probabilities
		first_place_counts_by_length = {length: {cand: 0.0 for cand in self.cand_names} for length in range(1, self.k + 1)}

		for idx, ballot in enumerate(ballots):
			first_place_counts_by_length[len(ballot)][ballot[0]] += ballot_counts[idx]

		if self.first_choice_prob_by_length:
			self.first_choice_prob = np.array([
				np.array([
					first_place_counts_by_length[length][self.cand_idx_to_id[idx]]
					for idx in range(self.k)
					])
				for length in range(1, self.k + 1)
				])
		else:
			shared_distribution = np.array([
				np.sum([first_place_counts_by_length[length][self.cand_idx_to_id[idx]] for length in range(1, self.k + 1)])
				for idx in range(self.k)])
			self.first_choice_prob = np.outer(np.ones(self.k), shared_distribution)

		# row normalize
		row_inv = [ 1 / max(np.sum(self.first_choice_prob[row_idx]), 1) for row_idx in range(self.k)]
		self.first_choice_prob = np.diag(row_inv) @ self.first_choice_prob
		# print(self.first_choice_prob)

		# length probabilities 
		length_counts = np.zeros(self.k + 1)
		for idx, ballot in enumerate(ballots):
			length_counts[len(ballot)] += ballot_counts[idx]
		self.length_prob = length_counts / np.sum(length_counts)
		# print(self.length_prob)

		# transition matrix
		for ballot_idx, ballot in enumerate(ballots):
			for prev_idx, cand_j in enumerate(ballot[1:]):
				cand_i = ballot[prev_idx]
				
				cand_j_idx = self.cand_id_to_idx[cand_j]
				cand_i_idx = self.cand_id_to_idx[cand_i]

				self.transition_mtx[cand_i_idx, cand_j_idx] += ballot_counts[ballot_idx]
		# print(self.transition_mtx)

		# row normalize
		diagonals = [np.sum(row) if np.sum(row) > 0 else 1 for row in self.transition_mtx]
		self.transition_mtx = np.linalg.inv(np.diag(diagonals)) @ self.transition_mtx

		self.is_fitted = True

	def simulate_ballots(self, num_ballots, seed=0):
		ctr = Counter()
		for _ in range(num_ballots):
			length = self.rng.choice(self.k+1, p=self.length_prob)

			prev_idx = self.rng.choice(self.k, p=self.first_choice_prob[length - 1])
			ballot = [prev_idx]

			all_cands = np.arange(self.k)
			available_cand_mask = np.ones(self.k) == 1
			available_cand_mask[prev_idx] = False

			# while (len(ballot) < self.k) and (prev_idx != self.k):
			while len(ballot) < length:
				possible_cands = all_cands[available_cand_mask]
				p = self.transition_mtx[prev_idx][available_cand_mask]
				if np.sum(p) == 0:
					p = None
				else:
					p /= np.sum(p)

				next_idx = self.rng.choice(
					possible_cands, 
					p=p)

				# handle sampled cand
				prev_idx = next_idx
				ballot.append(next_idx)
				available_cand_mask[next_idx] = False
			ballot_with_ids = tuple([self.cand_idx_to_id[idx] for idx in ballot])
			ctr[ballot_with_ids] += 1

		items = ctr.most_common()
		output_ballots = [np.array(tpl) for tpl, _ in items]
		ballot_counts = [cnt for _, cnt in items]

		assert np.all([len(ballot) > 0 for ballot in output_ballots])
		return output_ballots, ballot_counts

if __name__ == "__main__":
	burlington_filename = "data/preflib/elections-all/burlington/ED-00005-00000002.toi"
	ballots, ballot_counts, cand_names, skipped_votes = \
		utils.read_preflib(burlington_filename)

	num_ballots = np.sum(ballot_counts)

	models = [
		Bootstrap(cand_names),
		PLModel(cand_names),
		ContextModel(cand_names),
		ContextModel(cand_names, first_choice_prob_by_length=True)
	]

	for ballot_model in models:
		start = time.time()
		ballot_model.fit(ballots.copy(), ballot_counts)
		print(f"Fitting Time: {time.time() - start}")

		start = time.time()
		simulated_ballots, simulated_counts = ballot_model.simulate_ballots(num_ballots)
		print(f"Simulation Time: {time.time() - start}")

		for idx in range(min(len(simulated_ballots), 5)):
			print(f"{simulated_ballots[idx]}:\t{simulated_counts[idx]}")
		print("\n")


