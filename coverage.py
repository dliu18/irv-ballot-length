import model
import utils 

from multiprocessing import Pool, cpu_count
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt 

from tqdm import tqdm

NUM_TRIALS = 10

def get_coverage(trial_tuple):
	all_ballots, all_counts, \
	sample_ballots, sample_counts, \
	ballot_model, model_name, sample_ratio = trial_tuple

	n = np.sum(all_counts)	
	ballot_model.fit(sample_ballots.copy(), sample_counts)
	simulated_ballots, simulated_counts = ballot_model.simulate_ballots(n)

	ctr = Counter()
	for ballot_idx, ballot in enumerate(all_ballots):
		ctr[tuple(ballot)] = all_counts[ballot_idx]

	for ballot_idx, ballot in enumerate(simulated_ballots):
		if tuple(ballot) not in ctr:
			continue

		global_value = ctr[tuple(ballot)]
		ctr[tuple(ballot)] = max(global_value - simulated_counts[ballot_idx], 0)

	coverage = 1 - (np.sum([ctr[ballot] for ballot in ctr]) / n)

	# print(model_name)
	# print(f"Oracle Ballots:")
	# for idx, ballot in enumerate(all_ballots[:10]):
	# 	print(f"{ballot}: {all_counts[idx]}")

	# print("Simulated Ballots:")
	# for idx, ballot in enumerate(simulated_ballots[:10]):
	# 	print(f"{ballot}: {simulated_counts[idx]}")
	return model_name, sample_ratio, coverage

def get_recall(trial_tuple):
	all_ballots, all_counts, \
	sample_ballots, sample_counts, \
	ballot_model, model_name, sample_ratio = trial_tuple

	n = np.sum(all_counts)	
	ballot_model.fit(sample_ballots.copy(), sample_counts)
	simulated_ballots, simulated_counts = ballot_model.simulate_ballots(n)

	ctr = Counter()
	for ballot in all_ballots:
		ctr[tuple(ballot)] = 0

	for idx, ballot in enumerate(simulated_ballots):
		if tuple(ballot) not in ctr or simulated_counts[idx] == 0:
			continue
		ctr[tuple(ballot)] += 1

	recall = np.sum([count > 0 for ballot, count in ctr.items()]) / len(all_counts)
	
	# print(model_name)
	# for ballot, count in ctr.items():
	# 	print(f"{ballot}: {count}")
	# print("\n")
	return model_name, sample_ratio, recall


def plot_coverage(results, election_name, ci=0.95, ax=None, metric="Coverage"):

    # z-scores for common CIs; fallback to 1.96 if unknown
    Z_TABLE = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    z = Z_TABLE.get(ci, 1.959963984540054)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for model, ratio_to_vals in results.items():
        # Sort x-values for a proper line
        xs = sorted(ratio_to_vals.keys(), key=float)

        means, lowers, uppers = [], [], []
        for x in xs:
            vals = np.asarray(ratio_to_vals[x], dtype=float)
            n = np.sum(~np.isnan(vals)) if np.isnan(vals).any() else len(vals)

            if n == 0:
                m = lo = up = np.nan
            else:
                m = float(np.nanmean(vals))
                # sample std with ddof=1 when n>1; 0 when n==1
                std = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0
                se = std / np.sqrt(n) if n > 0 else np.nan
                lo, up = m - z * se, m + z * se

            means.append(m)
            lowers.append(lo)
            uppers.append(up)

        xs_arr = np.array(xs, dtype=float)
        ax.plot(xs_arr, means, marker="o", label=model)
        ax.fill_between(xs_arr, lowers, uppers, alpha=0.2)

    title=f"{metric} vs. Sample Ratio by Model"
    ax.set_xlabel("Sample ratio")
    ax.set_xscale("log")
    ax.set_ylabel(f"Average {metric}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Model", frameon=False)
    ax.margins(x=0.02)

    fig.savefig(f"plots/coverage/{metric}_{election_name}.pdf", bbox_inches="tight")

if __name__ == "__main__":

	# sample_ratios = [0.01, 0.02, 0.04, 0.08, 0.1, 0.16, 0.2, 0.32, 0.64, 0.95, 1.0]
	# sample_ratios = [0.01]
	sample_ratios = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.95, 1.0]

	model_names = [
		"Bootstrap",
		"PL",
		# "Contextual", 
		"Contextual By Length", 
		# "Contextual By Length and Smoothing 1",
		# "Contextual By Length and Smoothing 5",
		# "Mallows Dispersion 0.5",
		"Mallows Dispersion 1",
		# "Mallows Dispersion 5",
		# "Mallows Dispersion 10",
		"Contextual Perturbation Dispersion 0.5",
		"Contextual Perturbation Dispersion 1",
		"Contextual Perturbation Dispersion 5",
		"Uniform"
		]
	# model_names = ["Bootstrap"]

	filename = "data/preflib/elections-all/burlington/ED-00005-00000002.toi"
	# filename = "data/preflib/elections-all/sf/ED-00021-00000007.toi"

	ballots, ballot_counts, cand_names, skipped_votes = \
		utils.read_preflib(filename)
	n = np.sum(ballot_counts)

	trials = []
	for model_name in model_names:
		for sample_ratio in sample_ratios:
			sample_size = int(n * sample_ratio)
			# print(sample_size)
			for seed in range(NUM_TRIALS):
				sample_counts = utils.resample(ballot_counts,
					sample_size=sample_size,
					with_replacement=False,
					seed=seed)
				# print(sample_counts)
				trials.append((
					ballots.copy(), ballot_counts,
					ballots.copy(), sample_counts,
					utils.get_model_object(model_name, cand_names),
					model_name,
					sample_ratio
					))
	print(len(trials))

	results = {model: {sample_ratio: [] for sample_ratio in sample_ratios} for model in model_names}
	threads = cpu_count()
	# threads = 1
	with Pool(threads) as pool:
		for res in tqdm(pool.imap_unordered(get_coverage, trials), total=len(trials)):
			model_name, sample_ratio, coverage = res
			results[model_name][sample_ratio].append(coverage)

	# print(results)
	election_name = "burlington"
	plot_coverage(results, election_name=election_name, metric="Coverage")

	results = {model: {sample_ratio: [] for sample_ratio in sample_ratios} for model in model_names}
	threads = cpu_count()
	# threads = 1
	with Pool(threads) as pool:
		for res in tqdm(pool.imap_unordered(get_recall, trials), total=len(trials)):
			model_name, sample_ratio, coverage = res
			results[model_name][sample_ratio].append(coverage)

	# print(results)
	election_name = "burlington"
	plot_coverage(results, election_name=election_name, metric="Recall")
