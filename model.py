import utils 
import time

# import choix
import numpy as np 

from collections import Counter

from irv import run_irv

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

class UniformModel(BaseModel):
	def simulate_ballots(self, num_ballots, seed=0):
		assert self.is_fitted

		cands = list(self.cand_names.keys())

		# ballots = [self.rng.permutation(cands) for _ in range(num_ballots)]
		unshuffled_ballots = np.array([cands for _ in range(num_ballots)])
		shuffled_ballots = self.rng.permuted(unshuffled_ballots, axis=1)

		ctr = Counter()
		for ballot in shuffled_ballots:
			ctr[tuple(ballot)] += 1

		items = ctr.most_common()
		output_ballots = [np.array(tpl) for tpl, _ in items]
		ballot_counts = tuple([cnt for _, cnt in items])

		return output_ballots, ballot_counts

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
		return self.ballots.copy(), tuple(simulated_counts)

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
		ballot_counts = tuple([cnt for _, cnt in items])

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
	def __init__(self, 
			cand_names, 
			seed=0, 
			first_choice_prob_by_length=False,
			rank_weighted_first_choice=False,
			rank_weighted_transition_mtx=False,
			truncation_h=-1,
			smoothing=0):
		super().__init__(cand_names, seed)

		# probability each cand is the first choice in a ballot
		# includes the option to learn a separate distribution for each ballot length, where the length row indexes the matrix
		# however, by default, all ballot lengths share the same distribution.
		self.first_choice_prob = np.zeros((self.k, self.k))
		self.first_choice_prob_by_length = first_choice_prob_by_length 

		self.rank_weighted_first_choice = rank_weighted_first_choice
		self.rank_weighted_transition_mtx = rank_weighted_transition_mtx

		self.truncation_h = truncation_h

		self.length_prob = np.zeros(self.k + 1) # discrete prob dist over ballot lengths. P(length = 0) = 0
		self.transition_mtx = np.zeros((self.k, self.k))


		# smoothing parameter for laplace smoothing when estimating probabilities
		self.smoothing = smoothing

		# Mapping
		self.cand_id_to_idx = {cid: i for i, cid in enumerate(self.cand_names)}
		self.cand_idx_to_id = {i: cid for cid, i in self.cand_id_to_idx.items()}

	def get_params(self):
		return self.first_choice_prob.copy(), self.length_prob.copy(), self.transition_mtx.copy()

	def fit(self, ballots, ballot_counts):
		if self.truncation_h > 0:
			elim_votes = run_irv(
				self.k,
				ballots.copy(),
				ballot_counts,
				self.cand_names
				)
			truncated_cands = utils.get_elim_order(elim_votes)[-self.truncation_h:]
			ballots, ballot_counts = utils.reduce_election(ballots, ballot_counts, truncated_cands)

		# first probabilities
		first_place_counts_by_length = {length: {cand: 0.0 for cand in self.cand_names} for length in range(1, self.k + 1)}

		for idx, ballot in enumerate(ballots):
			if self.rank_weighted_first_choice:
				for cand_idx in range(len(ballot)):
					first_place_counts_by_length[len(ballot)][ballot[cand_idx]] +=  ballot_counts[idx] / (1 + cand_idx)
			else:
				first_place_counts_by_length[len(ballot)][ballot[0]] +=  ballot_counts[idx]

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
		self.first_choice_prob += self.smoothing

		row_inv = [ 1 / max(np.sum(self.first_choice_prob[row_idx]), 1) for row_idx in range(self.k)]
		self.first_choice_prob = np.diag(row_inv) @ self.first_choice_prob
		# print(self.first_choice_prob)

		# length probabilities 
		length_counts = np.zeros(self.k + 1)
		length_counts[1:] = self.smoothing

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

				if self.rank_weighted_transition_mtx:
					self.transition_mtx[cand_i_idx, cand_j_idx] += ballot_counts[ballot_idx] / (1 + prev_idx)
				else:
					self.transition_mtx[cand_i_idx, cand_j_idx] += ballot_counts[ballot_idx]

		# print(self.transition_mtx)

		# row normalize
		self.transition_mtx += self.smoothing
		diagonals = [np.sum(row) if np.sum(row) > 0 else 1 for row in self.transition_mtx]
		self.transition_mtx = np.linalg.inv(np.diag(diagonals)) @ self.transition_mtx

		self.is_fitted = True

	# def simulate_ballots(self, num_ballots, seed=0):
	# 	ctr = Counter()
	# 	for _ in range(num_ballots):
	# 		length = self.rng.choice(self.k+1, p=self.length_prob)

	# 		prev_idx = self.rng.choice(self.k, p=self.first_choice_prob[length - 1])
	# 		ballot = [prev_idx]

	# 		all_cands = np.arange(self.k)
	# 		available_cand_mask = np.ones(self.k) == 1
	# 		available_cand_mask[prev_idx] = False

	# 		# while (len(ballot) < self.k) and (prev_idx != self.k):
	# 		while len(ballot) < length:
	# 			possible_cands = all_cands[available_cand_mask]
	# 			p = self.transition_mtx[prev_idx][available_cand_mask]
	# 			if np.sum(p) == 0:
	# 				p = None
	# 			else:
	# 				p /= np.sum(p)

	# 			next_idx = self.rng.choice(
	# 				possible_cands, 
	# 				p=p)

	# 			# handle sampled cand
	# 			prev_idx = next_idx
	# 			ballot.append(next_idx)
	# 			available_cand_mask[next_idx] = False
	# 		ballot_with_ids = tuple([self.cand_idx_to_id[idx] for idx in ballot])
	# 		ctr[ballot_with_ids] += 1

	# 	items = ctr.most_common()
	# 	output_ballots = [np.array(tpl) for tpl, _ in items]
	# 	ballot_counts = tuple([cnt for _, cnt in items])

	# 	assert np.all([len(ballot) > 0 for ballot in output_ballots])
	# 	return output_ballots, ballot_counts

	def simulate_ballots(self, num_ballots, seed=0):
		"""
		Vectorized ballot simulation for ContextModel.

		- Uses self.length_prob over {0, ..., k} for ballot lengths
		- Uses self.first_choice_prob[length-1] for the first choice
		- Uses self.transition_mtx with a without-replacement mask for subsequent choices
		- Raises ValueError if any length-0 ballot is generated
		"""
		# You can add `assert self.is_fitted` if you want a safety check
		
		if num_ballots <= 0:
			return [], []

		rng = self.rng

		k = self.k
		length_prob = self.length_prob              # shape (k+1,)
		first_choice_prob = self.first_choice_prob  # shape (k, k) or (max_len, k)
		T = self.transition_mtx                     # shape (k, k)
		cand_ids = self.cand_idx_to_id              # dict: idx -> candidate ID

		# ---- 1. Sample lengths ---------------------------------------------------
		lengths = rng.choice(k + 1, size=num_ballots, p=length_prob)

		# Hard error if any length-0 ballot is generated
		if np.any(lengths == 0):
			raise ValueError(
				"simulate_ballots(): generated length-0 ballots, "
				"but length-0 ballots are not allowed.\n"
				"Hint: check self.length_prob — its first entry "
				"(probability of length=0) must be 0."
			)

		max_len = int(lengths.max())

		# ballots_idx: (num_ballots, max_len), -1 means "unused slot"
		ballots_idx = -np.ones((num_ballots, max_len), dtype=np.int64)
		# availability: (num_ballots, k)
		available = np.ones((num_ballots, k), dtype=bool)

		# ---- 2. First choices (grouped by length) --------------------------------
		unique_lengths = np.unique(lengths)
		first_choices = np.empty(num_ballots, dtype=np.int64)

		for L in unique_lengths:
			idx = np.where(lengths == L)[0]
			# row for ballots of length L
			p_first = first_choice_prob[L - 1]
			first_choices[idx] = rng.choice(k, size=idx.size, p=p_first)

		ballots_idx[:, 0] = first_choices
		available[np.arange(num_ballots), first_choices] = False

		# ---- 3. Remaining positions ---------------------------------------------
		for pos in range(1, max_len):
			# ballots that still need a candidate at this position
			active = lengths > pos
			if not np.any(active):
				break

			active_idx = np.where(active)[0]
			prev = ballots_idx[active_idx, pos - 1]

			# sanity: prev should always be >= 0 if lengths > pos
			if np.any(prev < 0):
				raise RuntimeError(
					"Internal error: encountered prev < 0 for a ballot with length > pos."
				)

			# Take transition rows for these previous candidates
			P = T[prev, :].copy()              # (num_active, k)
			mask = available[active_idx, :]    # (num_active, k)
			P *= mask                          # zero out unavailable

			row_sum = P.sum(axis=1, keepdims=True)
			has_mass = row_sum[:, 0] > 0

			# Normalize rows with nonzero mass
			if np.any(has_mass):
				P[has_mass] /= row_sum[has_mass]

			# Fallback: uniform among available candidates where row_sum == 0
			if np.any(~has_mass):
				mask_nm = mask[~has_mass]          # (num_zero_mass, k)
				denom = mask_nm.sum(axis=1, keepdims=True)
				if np.any(denom == 0):
					raise RuntimeError(
						"Cannot assign next candidate: some ballots have no "
						"available candidates left but still require more positions."
					)
				P[~has_mass] = mask_nm / denom

			# ---- Vectorized categorical sampling over rows of P ------------------
			u = rng.random(size=active_idx.size)       # (num_active,)
			cdf = np.cumsum(P, axis=1)                 # (num_active, k)
			next_idx = (cdf >= u[:, None]).argmax(axis=1)

			ballots_idx[active_idx, pos] = next_idx
			available[active_idx, next_idx] = False

		# ---- 4. Convert to candidate IDs + count ---------------------------------
		from collections import Counter
		ctr = Counter()

		for i in range(num_ballots):
			L = lengths[i]
			idxs = ballots_idx[i, :L]

			if np.any(idxs < 0):
				raise RuntimeError(
					f"Internal error: negative candidate index in ballot {i}: {idxs}"
				)

			# cand_ids is a dict: idx -> cand_id
			ballot_ids = tuple(cand_ids[int(j)] for j in idxs)
			ctr[ballot_ids] += 1

		items = ctr.most_common()
		output_ballots = [np.array(tpl) for tpl, _ in items]
		ballot_counts = tuple([cnt for _, cnt in items])

		return output_ballots, ballot_counts

class MallowsModel(BaseModel):
	def __init__(self, cand_names, dispersion=1.0, distance_fn="kendall", seed=0):
		"""
		Mallows model for simulating (partial) ranked-choice ballots.

		Parameters
		----------
		cand_names : dict
			Dictionary mapping candidate ids to full/display names.
		dispersion : float
			Dispersion parameter θ. Larger values => rankings closer to consensus.
		distance_fn : str
			Distance function name. Supported: "kendall", "footrule", "edit".
			Simulation is:
				- exact for "kendall" (via inversion-vector sampler),
				- approximate MCMC for "footrule" and "edit".
		seed : int
			Random seed.
		"""
		super().__init__(cand_names, seed)
		self.dispersion = dispersion
		self.distance_fn = distance_fn.lower()
		if self.distance_fn not in {"kendall", "footrule", "edit"}:
			raise ValueError("distance_fn must be one of {'kendall', 'footrule', 'edit'}.")

		self.ballots = None
		self.ballot_counts = None
		# Universe of candidate ids (used for edit-distance proposals)
		self.all_cands = list(cand_names.keys())

	def fit(self, ballots, ballot_counts):
		"""
		For this model, fitting just stores the observed ballots and counts.

		Parameters
		----------
		ballots : list[list]
			List of (partial) ballots. Each ballot is an ordered list of candidate ids.
		ballot_counts : sequence[int]
			Tuple/list where entry i is the number of occurrences of ballots[i].
		"""
		self.is_fitted = True
		self.ballots = [list(b) for b in ballots]
		self.ballot_counts = list(ballot_counts)

	def distance(self, ballot1, ballot2):
		"""
		Compute the distance between two ballots using the configured distance_fn.

		Parameters
		----------
		ballot1, ballot2 : sequence
			Partial rankings (lists/arrays) of candidate ids.

		Returns
		-------
		int
			Distance between ballot1 and ballot2.
		"""
		if self.distance_fn == "kendall":
			return utils.kendall_tau_distance_partial(ballot1, ballot2, self.all_cands)
		elif self.distance_fn == "footrule":
			return utils.footrule_distance_partial(ballot1, ballot2, self.all_cands)
		elif self.distance_fn == "edit":
			return utils.edit_distance_partial(ballot1, ballot2, self.all_cands)
		else:
			raise ValueError(f"Unsupported distance_fn '{self.distance_fn}'.")

	# ---------------------------------------------------------------------
	# Mallows sampling
	# ---------------------------------------------------------------------

	def _sample_mallows_ballot_kendall(self, consensus_ballot, theta, rng):
		"""
		Sample a single ballot from a Mallows model with Kendall distance
		centered at the given consensus partial ballot.

		The sampled ballot has the same candidate set and length as the consensus.
		This is an exact sampler based on the inversion-vector representation.
		"""
		m = len(consensus_ballot)
		if m <= 1 or theta is None:
			return list(consensus_ballot)

		# Draw independent "inversion counts" V_j ~ truncated geometric on {0..j-1}
		# with P(V_j=v) ∝ exp(-theta * v), j = 1..m.
		# We use 1-based indexing for clarity in this internal representation.
		V = [None] * (m + 1)
		V[1] = 0  # by definition

		for j in range(2, m + 1):
			if theta == 0:
				# Uniform over {0, ..., j-1}
				probs = np.ones(j) / float(j)
			else:
				exps = np.exp(-theta * np.arange(j))
				probs = exps / exps.sum()
			V[j] = rng.choice(np.arange(j), p=probs)

		# Convert inversion vector V to a permutation of positions [0..m-1]
		# using the standard construction for Kendall's tau Mallows:
		# start with [1], then for j=2..m insert j at position (j-1 - V_j).
		seq = [1]
		for j in range(2, m + 1):
			v = V[j]
			insert_pos = (j - 1 - v)  # 0-based index
			seq.insert(insert_pos, j)

		# seq now is a permutation of {1,...,m}; convert to 0-based indices
		perm_indices = [x - 1 for x in seq]

		# Apply permutation to consensus_ballot
		return [consensus_ballot[idx] for idx in perm_indices]

	def _propose_neighbor(self, ballot, rng):
		"""
		Propose a neighboring ballot for MCMC under the current distance_fn.

		For "kendall" and "footrule":
			- Propose by swapping two distinct positions (keeps length and candidate set).
		For "edit":
			- Randomly choose among:
				* swap two positions,
				* insert an unranked candidate at a random position,
				* delete a candidate at a random position
			  while preserving the constraint of no duplicate candidates and
			  non-empty ballots.
		"""
		if len(ballot) == 0:
			return list(ballot)

		b = list(ballot)

		if self.distance_fn in {"kendall", "footrule"}:
			if len(b) == 1:
				return b
			# simple random swap
			i, j = rng.integers(0, len(b), size=2)
			while j == i:
				j = rng.integers(0, len(b))
			b[i], b[j] = b[j], b[i]
			return b

		# distance_fn == "edit"
		move_type = rng.choice(["swap", "insert", "delete"])

		if move_type == "swap" or len(b) == 1:
			# swap two positions
			if len(b) == 1:
				return b
			i, j = rng.integers(0, len(b), size=2)
			while j == i:
				j = rng.integers(0, len(b))
			b[i], b[j] = b[j], b[i]
			return b

		elif move_type == "insert":
			# insert an unranked candidate at random position (if any available)
			available = list(set(self.all_cands) - set(b))
			if not available:
				return b  # no new candidates to insert
			cand = rng.choice(available)
			pos = rng.integers(0, len(b) + 1)
			b.insert(pos, cand)
			return b

		elif move_type == "delete":
			# delete a random candidate, but keep at least one
			if len(b) == 1:
				return b
			pos = rng.integers(0, len(b))
			del b[pos]
			return b

		return b

	def _sample_mallows_ballot_mcmc(self, consensus_ballot, theta, rng, n_steps=50):
		"""
		Approximate sampler for Mallows models with non-Kendall distances
		("footrule" and "edit") using Metropolis–Hastings MCMC.

		Parameters
		----------
		consensus_ballot : list
			Consensus (center) ballot.
		theta : float
			Dispersion parameter.
		rng : np.random.Generator
			Random number generator.
		n_steps : int
			Number of MCMC steps starting from the consensus.

		Returns
		-------
		list
			Sampled ballot approximately distributed according to
			P(b) ∝ exp(-theta * distance(b, consensus_ballot)).
		"""
		if theta is None:
			theta = 0.0

		current = list(consensus_ballot)
		current_d = self.distance(current, consensus_ballot)

		if n_steps <= 0 or theta < 0:
			return current

		for _ in range(n_steps):
			proposal = self._propose_neighbor(current, rng)
			if proposal == current:
				continue

			prop_d = self.distance(proposal, consensus_ballot)
			delta = prop_d - current_d

			if delta <= 0:
				accept = True
			else:
				accept = rng.random() < np.exp(-theta * delta)

			if accept:
				current, current_d = proposal, prop_d

		return current

	def _sample_mallows_ballot(self, consensus_ballot, theta, rng):
		"""
		Generic wrapper for sampling a ballot from the Mallows model.

		- If distance_fn == "kendall": use exact Kendall-based sampler.
		- If distance_fn in {"footrule", "edit"}: use MCMC approximate sampler.
		"""
		if self.distance_fn == "kendall":
			return self._sample_mallows_ballot_kendall(consensus_ballot, theta, rng)
		else:
			return self._sample_mallows_ballot_mcmc(consensus_ballot, theta, rng)

	def simulate_ballots(self, num_ballots, seed=0):
		"""
		Simulate ballots from a mixture of Mallows models, one centered at each
		observed consensus ballot. The number of samples drawn from each
		consensus is proportional to its prevalence in the fitted data.

		Parameters
		----------
		num_ballots : int
			Total number of ballots to simulate.
		seed : int
			Random seed for the simulation.

		Returns
		-------
		simulated_ballots : list[np.ndarray]
			List of unique simulated ballots (each as an array of candidate ids).
		simulated_counts : list[int]
			Counts corresponding to each simulated ballot.
		"""
		assert self.is_fitted, "Model must be fit before simulation."

		if num_ballots <= 0:
			return [], []

		if self.ballots is None or len(self.ballots) == 0:
			raise ValueError("No ballots stored in model. Fit must be called with non-empty ballots.")

		rng = np.random.default_rng(seed)

		# Probabilities over consensus ballots proportional to their observed counts
		counts_arr = np.array(self.ballot_counts, dtype=float)
		total = counts_arr.sum()
		if total <= 0:
			raise ValueError("Total ballot count must be positive.")
		probs = counts_arr / total

		# Number of simulated ballots from each consensus
		sim_counts_per_consensus = rng.multinomial(num_ballots, probs)

		ctr = Counter()
		for idx, consensus_ballot in enumerate(self.ballots):
			n_samples = sim_counts_per_consensus[idx]
			if n_samples == 0:
				continue

			for _ in range(n_samples):
				sampled = self._sample_mallows_ballot(consensus_ballot, self.dispersion, rng)
				if len(sampled) == 0:
					continue  # ignore pathological empty ballots
				ctr[tuple(sampled)] += 1

		# Convert Counter to lists of ballots and counts
		items = ctr.most_common()
		simulated_ballots = [np.array(ballot_tuple) for ballot_tuple, _ in items]
		simulated_counts = tuple([cnt for _, cnt in items])

		assert np.all([len(b) > 0 for b in simulated_ballots])

		return simulated_ballots, simulated_counts

class ContextualPerturbationModel(BaseModel):
	def __init__(self, cand_names, dispersion=1.0, seed=0, smoothing=0.0):
		"""
		Model that perturbs observed (consensus) ballots by shortening or extending
		ballots using a discretized normal perturbation and a learned transition
		matrix (same construction as ContextModel).

		Parameters
		----------
		cand_names : dict
			Dictionary mapping candidate ids to full/display names.
		dispersion : float
			Variance parameter for the discretized normal perturbation. Larger values
			make larger positive/negative changes in ballot length more likely.
		seed : int
			Random seed.
		smoothing : float
			Laplace smoothing added before row-normalizing the transition matrix.
		"""
		super().__init__(cand_names, seed)
		self.dispersion = dispersion
		self.smoothing = float(smoothing)

		# Store fitted ballots and counts
		self.ballots = None
		self.ballot_counts = None

		# Transition matrix over candidate indices (same size/role as in ContextModel)
		self.transition_mtx = np.zeros((self.k, self.k))

		# Candidate ID <-> index mappings
		self.cand_id_to_idx = {cid: i for i, cid in enumerate(self.cand_names)}
		self.cand_idx_to_id = {i: cid for cid, i in self.cand_id_to_idx.items()}

		# Cache list of all candidate indices for convenience
		self._all_indices = list(range(self.k))

	def fit(self, ballots, ballot_counts):
		"""
		Fit the model by storing ballots/counts and learning the transition matrix.

		The transition matrix is learned in the same way as ContextModel:
		for each adjacent pair (c_i, c_j) observed in a ballot, we increment
		T[i, j] by the ballot count. Then we add Laplace smoothing and row-normalize.
		"""
		self.is_fitted = True
		self.ballots = [list(b) for b in ballots]
		self.ballot_counts = list(ballot_counts)

		# Reset transition matrix
		self.transition_mtx.fill(0.0)

		# Accumulate transitions
		for ballot_idx, ballot in enumerate(self.ballots):
			count = self.ballot_counts[ballot_idx]
			if count <= 0:
				continue
			# For each adjacent pair (cand_i -> cand_j)
			for prev_pos, cand_j in enumerate(ballot[1:]):
				cand_i = ballot[prev_pos]
				# Map candidate IDs to indices
				if cand_i not in self.cand_id_to_idx or cand_j not in self.cand_id_to_idx:
					continue
				cand_i_idx = self.cand_id_to_idx[cand_i]
				cand_j_idx = self.cand_id_to_idx[cand_j]
				self.transition_mtx[cand_i_idx, cand_j_idx] += count

		# Row-normalize with smoothing (same pattern as ContextModel)
		self.transition_mtx += self.smoothing
		diagonals = [np.sum(row) if np.sum(row) > 0 else 1.0 for row in self.transition_mtx]
		self.transition_mtx = np.linalg.inv(np.diag(diagonals)) @ self.transition_mtx

	def _sample_discrete_normal(self, rng):
		"""
		Sample an integer from a discretized normal N(0, dispersion).

		We draw a real-valued normal and round it to the nearest integer.
		If dispersion == 0, this always returns 0.
		"""
		if self.dispersion <= 0:
			return 0
		z = rng.normal(loc=0.0, scale=1/self.dispersion)
		return int(np.round(z))

	def _extend_ballot_with_transitions(self, ballot, num_add, rng):
		"""
		Extend a ballot by `num_add` candidates using the transition matrix.

		We treat the transitions as a Markov chain over candidate indices:
		at each step, we look at the last candidate in the ballot, then sample
		the next candidate according to the corresponding row of the transition
		matrix, with previously chosen candidates masked out (without replacement).
		If masking yields a zero row, we fall back to a uniform distribution over
		remaining candidates.
		"""
		if num_add <= 0:
			return ballot

		# Work on indices for ease with transition_mtx
		idx_ballot = [self.cand_id_to_idx[cid] for cid in ballot]
		chosen = set(idx_ballot)

		for _ in range(num_add):
			# Determine available candidates (not yet in ballot)
			available = [idx for idx in self._all_indices if idx not in chosen]
			if not available:
				break  # no more candidates to add

			last_idx = idx_ballot[-1]
			row = self.transition_mtx[last_idx].copy()

			# Mask out already-chosen candidates
			for idx in chosen:
				row[idx] = 0.0

			row_sum = row.sum()
			if row_sum <= 0:
				# Fall back to uniform over remaining
				next_idx = rng.choice(available)
			else:
				row /= row_sum
				# Only allow choices in `available`
				probs_available = np.array([row[idx] for idx in available], dtype=float)
				prob_sum = probs_available.sum()
				if prob_sum <= 0:
					next_idx = rng.choice(available)
				else:
					probs_available /= prob_sum
					next_idx = rng.choice(available, p=probs_available)

			idx_ballot.append(next_idx)
			chosen.add(next_idx)

		# Convert back to candidate IDs
		return [self.cand_idx_to_id[idx] for idx in idx_ballot]

	def _perturb_ballot(self, consensus_ballot, rng):
		"""
		Perturb a single consensus ballot according to the dispersion rule.

		1. Draw D ~ discretized N(0, dispersion).
		2. If D == 0: return the consensus ballot unchanged.
		3. If D < 0: delete -D candidates from the end of the ballot, but always
		   leave at least one candidate.
		4. If D > 0: extend the ballot by D candidates using the transition matrix.
		"""
		ballot = list(consensus_ballot)
		if not ballot:
			return ballot

		delta = self._sample_discrete_normal(rng)

		if delta == 0:
			return ballot

		if delta < 0:
			# ensure we keep at least one candidate
			num_delete = min(-delta, max(len(ballot) - 1, 0))
			if num_delete > 0:
				ballot = ballot[:-num_delete]
			return ballot

		# delta > 0: add candidates
		return self._extend_ballot_with_transitions(ballot, delta, rng)

	def simulate_ballots(self, num_ballots, seed=0):
		"""
		Simulate ballots by perturbing fitted consensus ballots.

		The mixture over consensus ballots is proportional to their observed counts.
		"""
		assert self.is_fitted, "Model must be fit before simulation."

		if num_ballots <= 0:
			return [], []

		if self.ballots is None or len(self.ballots) == 0:
			raise ValueError("No ballots stored in model. Fit must be called with non-empty ballots.")

		# Use a fresh RNG for simulation, similar to MallowsModel
		rng = np.random.default_rng(seed)

		counts_arr = np.array(self.ballot_counts, dtype=float)
		total = counts_arr.sum()
		if total <= 0:
			raise ValueError("Total ballot count must be positive.")
		probs = counts_arr / total

		# Number of simulated ballots from each consensus
		sim_counts_per_consensus = rng.multinomial(num_ballots, probs)

		ctr = Counter()
		for idx, consensus_ballot in enumerate(self.ballots):
			n_samples = sim_counts_per_consensus[idx]
			if n_samples == 0:
				continue

			for _ in range(n_samples):
				perturbed = self._perturb_ballot(consensus_ballot, rng)
				if len(perturbed) == 0:
					continue  # ignore pathological empty ballots
				ctr[tuple(perturbed)] += 1

		items = ctr.most_common()
		output_ballots = [np.array(list(tpl)) for tpl, _ in items]
		ballot_counts = tuple([cnt for _, cnt in items])

		assert np.all([len(ballot) > 0 for ballot in output_ballots])

		return output_ballots, ballot_counts


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def visualize_ballot_embeddings(
    seed_ballots,
    seed_counts,
    sim_ballots,
    sim_counts,
    distance_fn,
    laplace_scale=1.0,
    random_state=0
):
    """
    Visualize seed vs simulated ballots via Laplacian Eigenmaps on a
    Laplace-kernel-weighted ballot similarity graph.

    Parameters
    ----------
    seed_ballots : list[list] or list[tuple]
        List of seed ballot types. Each ballot is an ordered iterable of candidate IDs.
    seed_counts : sequence[int]
        Counts for each seed ballot type (same length as seed_ballots).
    sim_ballots : list[list] or list[tuple]
        List of simulated ballot types.
    sim_counts : sequence[int]
        Counts for each simulated ballot type (same length as sim_ballots).
    distance_fn : callable
        Function distance_fn(b1, b2) -> float
        that returns a (nonnegative) distance between two ballots.
        (E.g., Kendall / Footrule / edit distance over partial rankings.)
    laplace_scale : float, default=1.0
        Scale parameter λ for the Laplace kernel:
            w_ij = exp( - d_ij / λ )
        Larger λ → slower decay with distance.
    random_state : int, default=0
        Random seed used only to break eigenvector sign / ordering ties in
        a reproducible way (numpy.linalg.eigh is deterministic, but we set
        the RNG for safety if any randomness is added later).

    Returns
    -------
    embeddings : np.ndarray, shape (n_ballot_types, 2)
        2D embedding coordinates for all ballot types (seed first, then simulated).
    is_seed : np.ndarray, shape (n_ballot_types,)
        Boolean mask indicating which rows correspond to seed ballots.
    """

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # 1. Combine all ballot types & counts
    # ------------------------------------------------------------------
    seed_ballots = [list(b) for b in seed_ballots]
    sim_ballots = [list(b) for b in sim_ballots]

    seed_counts = np.asarray(seed_counts, dtype=float)
    sim_counts = np.asarray(sim_counts, dtype=float)

    all_ballots = seed_ballots + sim_ballots
    all_counts = np.concatenate([seed_counts, sim_counts], axis=0)

    n_seed = len(seed_ballots)
    n_sim = len(sim_ballots)
    n = n_seed + n_sim

    if n == 0:
        raise ValueError("No ballot types provided.")

    is_seed = np.zeros(n, dtype=bool)
    is_seed[:n_seed] = True

    # ------------------------------------------------------------------
    # 2. Compute pairwise distance matrix among all ballot types
    # ------------------------------------------------------------------
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = float(distance_fn(all_ballots[i], all_ballots[j]))
            D[i, j] = d_ij
            D[j, i] = d_ij

    # ------------------------------------------------------------------
    # 3. Convert distances to weights via Laplace kernel
    #     w_ij = exp( -d_ij / laplace_scale ), w_ii = 0
    # ------------------------------------------------------------------
    if laplace_scale <= 0:
        raise ValueError("laplace_scale must be positive.")

    W = np.exp(-D / laplace_scale)
    np.fill_diagonal(W, 0.0)  # no self-loops

    # ------------------------------------------------------------------
    # 4. Build graph Laplacian and compute Laplacian Eigenmaps
    # ------------------------------------------------------------------
    # Degree matrix
    degrees = W.sum(axis=1)
    # Handle isolated nodes by giving them degree 1 (so Laplacian is defined)
    degrees_safe = np.where(degrees > 0, degrees, 1.0)
    D_deg = np.diag(degrees_safe)

    # Unnormalized Laplacian
    L = D_deg - W

    # Eigen-decomposition of L
    # We use the smallest nontrivial eigenvectors as embedding dimensions.
    eigvals, eigvecs = np.linalg.eigh(L)
    # Sort by eigenvalues ascending
    idx_sorted = np.argsort(eigvals)
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    # The first eigenvector is (approximately) constant; skip it.
    # Take the next two for a 2D embedding.
    if n < 3:
        # degenerate case: just take whatever is available
        embed_dims = min(2, n)
        X = eigvecs[:, 1:1 + embed_dims]
        # pad to 2D if needed
        if embed_dims == 1:
            X = np.hstack([X, np.zeros((n, 1))])
    else:
        X = eigvecs[:, 1:3]

    embeddings = X  # shape (n, 2)

    # ------------------------------------------------------------------
    # 5. Prepare colors with opacity based on ballot counts
    # ------------------------------------------------------------------
    counts = all_counts
    if np.all(counts == counts[0]):
        # all counts equal: constant opacity
        alphas = np.ones_like(counts, dtype=float)
    else:
        c_min = counts.min()
        c_max = counts.max()
        # linear scaling to [0.2, 1.0] for visibility
        alphas = 0.2 + 0.8 * (counts - c_min) / (c_max - c_min)

    # Base colors: blue for seed, orange for simulated
    base_seed = np.array([0.121, 0.466, 0.705])  # matplotlib tab:blue
    base_sim = np.array([0.843, 0.373, 0.000])   # matplotlib tab:orange-ish

    colors = np.zeros((n, 4), dtype=float)
    for i in range(n):
        if is_seed[i]:
            rgb = base_seed
        else:
            rgb = base_sim
        colors[i, :3] = rgb
        colors[i, 3] = alphas[i]

    # ------------------------------------------------------------------
    # 6. Plot the embeddings
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=colors,
        s=40,
        edgecolors='none'
    )

    ax.set_xlabel("Laplacian eigenmap dim 1")
    ax.set_ylabel("Laplacian eigenmap dim 2")
    ax.set_title("Ballot-type Laplacian Eigenmap Embedding\n(Laplace kernel on distances)")

    # Legend (using dummy handles)
    seed_handle = Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor=base_seed,
        markersize=8,
        label='Seed ballots'
    )
    sim_handle = Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor=base_sim,
        markersize=8,
        label='Simulated ballots'
    )
    ax.legend(handles=[seed_handle, sim_handle], loc='best')

    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    return embeddings, is_seed

if __name__ == "__main__":
	# burlington_filename = "data/preflib/elections-all/burlington/ED-00005-00000002.toi"
	burlington_filename = "data/preflib/elections-all/sf/ED-00021-00000007.toi"
	
	ballots, ballot_counts, cand_names, skipped_votes = \
		utils.read_preflib(burlington_filename)

	num_ballots = np.sum(ballot_counts)

	models = [
		Bootstrap(cand_names),
		# PLModel(cand_names),
		ContextModel(cand_names),
		# ContextModel(cand_names, truncation_h=4),
		# ContextModel(cand_names, truncation_h=4, rank_weighted_first_choice=True),
		# ContextModel(cand_names, truncation_h=4, rank_weighted_transition_mtx=True),
		# ContextModel(cand_names, first_choice_prob_by_length=True),
		# ContextModel(cand_names, first_choice_prob_by_length=True, smoothing=1),
		MallowsModel(cand_names, distance_fn="kendall", dispersion=1),
		# MallowsModel(cand_names, distance_fn="footrule"),		
		# MallowsModel(cand_names, distance_fn="edit"),
		ContextualPerturbationModel(cand_names, dispersion=1),
		# UniformModel(cand_names)
	]

	for ballot_model in models:
		start = time.time()
		ballot_model.fit(ballots.copy(), ballot_counts)
		print(f"Fitting Time: {time.time() - start}")

		start = time.time()
		simulated_ballots, simulated_counts = ballot_model.simulate_ballots(num_ballots)
		print(f"Simulation Time: {time.time() - start}")

		# for idx in range(min(len(simulated_ballots), 5)):
		# 	print(f"{simulated_ballots[idx]}:\t{simulated_counts[idx]}")
		# print("\n")

		elim_votes = run_irv(len(cand_names), simulated_ballots.copy(), simulated_counts, cands=cand_names)
		simulated_winner = max(elim_votes, key=elim_votes.get)

		# print(f"Simulated Winner: {simulated_winner}")

