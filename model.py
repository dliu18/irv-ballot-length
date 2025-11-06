import utils 

import choix
import numpy as np 

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

		self.cand_id_to_idx = {cand_id:idx for idx, cand_id in enumerate(cand_names)}
		self.cand_id_to_idx["END"] = self.num_cands

		self.cand_idx_to_id = {idx: cand_id for idx, cand_id in enumerate(cand_names)}
		self.cand_idx_to_id[self.num_cands] = "END"


	def fit(self, ballots, ballot_counts):
		'''
			ballots is a list of ballots where each ballot is a list of ranked candidates.
			ballot_counts specifies the number of people ascribing to a particular ballot
		'''
		self.is_fitted=True


	def simulate_ballots(self, num_ballots):
		assert self.is_fitted


class PLModel(BaseModel):
	def __init__(self, cand_names, seed=0):
		super().__init__(cand_names, seed)

	def _convert_ballots_to_ranking_data(self, ballots, ballot_counts):
		ranking_data = []
		for ballot_idx, ballot_with_ids in enumerate(ballots):
			### convert to indices
			ballot_with_idxs = [
				self.cand_id_to_idx[cand_id] 
				for cand_id in ballot_with_ids
			]
			ballot_with_idxs.append(len(cand_names)) # end

			### handle unlisted candidates
			unlisted_cands = []
			for cand_idx in range(self.num_cands):
				if cand_idx not in ballot_with_idxs:
					unlisted_cands.append(cand_idx)
			
			for _ in range(ballot_counts[ballot_idx]):
				full_ballot = ballot_with_idxs.copy()
				if len(unlisted_cands) > 0:
					self.rng.shuffle(unlisted_cands)
					full_ballot.extend(unlisted_cands)
					
				assert len(full_ballot) == self.num_cands + 1
				ranking_data.append(full_ballot)

		assert len(ranking_data) == np.sum(ballot_counts)
		return ranking_data


	def _convert_choix_ballots_for_output(self, simulated_ballots):
		# Truncate the ballots after the end character
		ballots_and_counts = {}
		for ballot in simulated_ballots:
			end_idx = np.argwhere(ballot == self.num_cands)
			assert len(end_idx) == 1

			end_idx = end_idx[0][0] #extract the index
			if end_idx == 0:
				continue
				
			truncated_ballot = ballot[:end_idx]
			truncated_ballot_str = " ".join([str(cand) for cand in truncated_ballot])
			
			if truncated_ballot_str not in ballots_and_counts:
				ballots_and_counts[truncated_ballot_str] = 0
			ballots_and_counts[truncated_ballot_str] += 1

		# Sort by ballot counts and convert to output ballot format 
		sorted_ballots_and_counts = sorted(ballots_and_counts.items(), key=lambda x: -x[1])

		# Extract sorted keys and values
		output_ballots = []
		ballot_counts = []
		for ballot_str, count in sorted_ballots_and_counts:
			output_ballot = [self.cand_idx_to_id[int(cand_idx)] for cand_idx in ballot_str.split(" ")]
			output_ballots.append(output_ballot)
			ballot_counts.append(count)

		return output_ballots, ballot_counts
		 

	def fit(self, ballots, ballot_counts):
		self.is_fitted=True

		ranking_data = self._convert_ballots_to_ranking_data(ballots, ballot_counts)
		PL_params = choix.lsr.lsr_rankings(
			n_items=self.num_cands + 1,
			data=ranking_data
		)
		self.params = PL_params
		print(
			f"PL Params: \n", 
			"\n".join([
				f"{self.cand_names[self.cand_idx_to_id[cand_idx]]}:\t{weight}"
				for cand_idx, weight in enumerate(PL_params[:-1])
				])
			)

	def simulate_ballots(self, num_ballots):
		assert self.is_fitted
		
		simulated_ballots_from_choix = choix.utils.generate_rankings(
			self.params, 
			n_rankings=num_ballots, 
			size=self.num_cands + 1)

		return self._convert_choix_ballots_for_output(simulated_ballots_from_choix)

if __name__ == "__main__":
	burlington_filename = "data/preflib/elections-all/burlington/ED-00005-00000002.toi"
	ballots, ballot_counts, cand_names, skipped_votes = \
		utils.read_preflib(burlington_filename)

	num_ballots = np.sum(ballot_counts)

	plmodel = PLModel(cand_names)
	plmodel.fit(ballots, ballot_counts)
	simulated_ballots, simulated_counts = plmodel.simulate_ballots(num_ballots)

	print(simulated_ballots[:10], simulated_counts[:10])