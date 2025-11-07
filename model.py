import utils 

import choix
import numpy as np 

from collections import Counter

class BaseModel:
	def __init__(self, cand_names, seed=0):
		'''
			cand_names is a dictionary of candidate ids to full names
		'''
		self.is_fitted=False
		self.cand_names=cand_names
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
		self.ballot_counts = ballot_counts.copy()

	def simulate_ballots(self, num_ballots, seed=0): 
		simulated_counts = utils.resample(self.ballot_counts,
						sample_size=num_ballots,
						with_replacement=True,
						seed=seed)
		return self.ballots, simulated_counts

class PLModel(BaseModel):
	def __init__(self, cand_names, seed=0, use_end_marker=True, include_unranked=True):
		super().__init__(cand_names, seed)
		self.use_end_marker = use_end_marker
		self.include_unranked = include_unranked

		if self.use_end_marker:
			self.cand_names["END"] = ["END"]

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
		 
	def _convert_choix_ballots_for_output(self, simulated_ballots):
		ctr = Counter()
		for ballot in simulated_ballots:
			if self.use_end_marker:
				# find END index, skip if at front
				end_idx_arr = np.where(ballot == self.cand_id_to_idx["END"])[0]
				if len(end_idx_arr) == 0:
					end_idx = len(ballot)
				else:
					end_idx = int(end_idx_arr[0])
					if end_idx == 0:
						continue
			else:
				# no END marker, use full ballot
				end_idx = len(ballot)
			truncated = tuple(map(int, ballot[:end_idx])) #mapping to int ensure the tuple is hashable
			ctr[truncated] += 1

		# Sort by count desc
		items = ctr.most_common()
		output_ballots = [[self.cand_idx_to_id[i] for i in tpl] for tpl, _ in items]
		ballot_counts = [cnt for _, cnt in items]
		return output_ballots, ballot_counts
		
	def fit(self, ballots, ballot_counts, alpha=1e-3):
		self.is_fitted=True

		ranking_data = self._convert_ballots_to_ranking_data(ballots, ballot_counts)
		PL_params = choix.lsr.ilsr_rankings(
			n_items=self.num_cands + 1 if self.use_end_marker else self.num_cands,
			data=ranking_data,
			alpha=alpha
		)
		self.params = PL_params
		print(
			f"PL Params: \n", 
			"\n".join([
				f"{self.cand_names[self.cand_idx_to_id[cand_idx]]}:\t{weight}"
				for cand_idx, weight in enumerate(PL_params)
				])
			)

	def simulate_ballots(self, num_ballots, seed=0):
		assert self.is_fitted
		
		#generate_rankings does not have a seed parameter
		simulated_ballots_from_choix = choix.utils.generate_rankings(
			self.params, 
			n_rankings=num_ballots, 
			size=self.num_cands + 1 if self.use_end_marker else self.num_cands)

		return self._convert_choix_ballots_for_output(simulated_ballots_from_choix)

if __name__ == "__main__":
	burlington_filename = "data/preflib/elections-all/burlington/ED-00005-00000002.toi"
	ballots, ballot_counts, cand_names, skipped_votes = \
		utils.read_preflib(burlington_filename)

	num_ballots = np.sum(ballot_counts)

	plmodel = PLModel(cand_names,
		use_end_marker=True,
		include_unranked=True)
	plmodel.fit(ballots, ballot_counts)
	simulated_ballots, simulated_counts = plmodel.simulate_ballots(num_ballots)

	print(simulated_ballots[:10], simulated_counts[:10])