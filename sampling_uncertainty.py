import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 

from irv import run_irv
import utils
from model import Bootstrap, PLModel

import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import pickle

import argparse

# Before: (NUM_POLLS^2 + NUM_POLLS) * NUM_FORECASTING_TRIALS = 5000 
# After: 2 * NUM_POLLS * NUM_FORECASTING_TRIALS = 3200
NUM_FORECASTING_TRIALS = 5
NUM_POLLS = 3
NUM_FORECASTER_STDS = 3


def f(seed_counts, seed_ballots, n, cands, actual_winner, 
	n_trials=NUM_FORECASTING_TRIALS,
	sampler="Bootstrap"):
	'''
		Given a set of ballot counts from a poll and a known election size, calculate the probability that candidate "cand"
		wins across bootstrap trials, where in each trial the remaining ballots are sampled from seed_counts
	'''
	actual_winner_wins_counts = 0
	if sampler == "Bootstrap":
		sampling_model = Bootstrap(cands).fit(seed_ballots, seed_counts)
	elif sampler == "PL":
		sampling_model = PLModel(cands, use_end_marker=True, include_unranked=True)
		sampling_model.fit(seed_ballots, seed_counts)

	for seed in range(n_trials):
		simulated_ballots, simulated_counts = sampling_model.simulate_ballots(
			num_ballots=n-np.sum(seed_counts),
			seed=seed)
		simulated_ballots.extend(seed_ballots)
		simulated_counts.extend(seed_counts)
		
		assert np.sum(simulated_counts) == n

		elim_votes = run_irv(len(cands), simulated_ballots.copy(), simulated_counts, cands=cands)
		simulated_winner = max(elim_votes, key=elim_votes.get)
		actual_winner_wins_counts += int(simulated_winner == actual_winner)
	return actual_winner_wins_counts / n_trials


def plot_forecaster_vs_oracle(sampling_rates,
	oracle_mean_by_sampling_rate,
	oracle_std_by_sampling_rate,
	forecaster_mean_by_sampling_method,
	forecaster_std_by_sampling_method,
	fig_path,
	fig_name):
	
	fig, ax = plt.subplots(ncols=2, figsize=(10,5))

	# MEANS
	ax[0].plot(sampling_rates, oracle_mean_by_sampling_rate, color="black", linestyle="--", label="oracle")
	for forecaster_idx, (forecaster_name, forecaster_mean_by_sampling_rate) in enumerate(forecaster_mean_by_sampling_method.items()):
		ax[0].plot(
			x=sampling_rates,
			y=forecaster_mean_by_sampling_rate,
			color=utils.colors[forecaster_idx],
			linestyle="--",
			label=forecaster_name)
	ax[0].set_xlabel("Sampling Rate")
	ax[0].set_ylabel("Forecast Probability that Actual Winner Wins")
	ax[0].set_title(f"{fig_name} Forecast")
	ax[0].legend()

	# STD DEV
	ax[1].plot(sampling_rates, oracle_std_by_sampling_rate, color="black", linestyle="--", label="oracle")
	
	for forecaster_idx, (forecaster_name, forecaster_std_by_sampling_rate) in enumerate(forecaster_std_by_sampling_method.items()):
		xs, ys = [], []
		for idx, sampling_rate in enumerate(sampling_rates):
			xs.extend([sampling_rate] * len(forecaster_std_by_sampling_rate[idx]))
			ys.extend(forecaster_std_by_sampling_rate[idx])
		ax[1].scatter(
			x=xs,
			y=ys,
			alpha=0.5,
			color=utils.colors[forecaster_idx],
			label=forecaster_name)

	ax[1].set_xlabel("Sampling Rate")
	ax[1].set_ylabel("Std. Dev. Among Winner Forecasts")
	ax[1].set_title(f"{fig_name} Uncertainty")
	ax[1].legend()

	fig.savefig(f"{fig_path}/{fig_name}.pdf", bbox_inches="tight")


def forecaster_vs_oracle(ballots, ballot_counts, cand_names, actual_winner):
	sampling_rates = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.95, 1.0]
	samplers = ["Bootstrap", "PL"]
	forecasters = ["PL"]

	oracle_mean_by_sampling_rate = [] #one float entry per sampling ratio
	forecaster_mean_by_sampling_rate = {forecaster: [] for forecaster in forecasters}

	oracle_std_by_sampling_rate = [] #one float entry per sampling ratio
	forecaster_std_by_sampling_method = {sampler: [[] for _ in range(len(sampling_rates))] for sampler in samplers}

	for idx, sampling_rate in enumerate(sampling_rates):	
		oracle_probabilities = []
		forecaster_probabilities = {forecaster: [] for forecaster in forecaster_probabilities}
		
		# ORACLE UNCERTAINTY
		for oracle_seed in range(NUM_POLLS):
			poll_size = max(int(np.sum(ballot_counts) * sampling_rate), 1)
			oracle_poll = utils.resample(ballot_counts, 
				sample_size=poll_size, 
				with_replacement=False,
				seed=oracle_seed+10)

			# Calculate the probability the actual winner wins 
			oracle_prob_actual_winner_wins = f(
				seed_counts=oracle_poll, 
				seed_ballots=ballots,
				n=np.sum(ballot_counts), 
				cands=cand_names,
				actual_winner=actual_winner,
				sampler="Bootstrap")
			oracle_probabilities.append(oracle_prob_actual_winner_wins)
		
			for forecaster in forecasters:
				forecaster_prob_actual_winner_wins = f(
					seed_counts=oracle_poll, 
					seed_ballots=ballots,
					n=np.sum(ballot_counts), 
					cands=cand_names,
					actual_winner=actual_winner,
					sampler="PL")
				forecaster_probabilities[forecaster].append(oracle_prob_actual_winner_wins)

			if oracle_seed >= NUM_FORECASTER_STDS:
				continue

			# FORECASTER UNCERTAINTY
			for sampler in samplers:
				win_probabilities_across_seeds = [oracle_prob_actual_winner_wins]
				if sampler == "Bootstrap":
					sampling_model = Bootstrap(cands_names).fit(ballots, oracle_poll)
				elif sampler == "PL":
					sampling_model = PLModel(cands_names, use_end_marker=True, include_unranked=True)
					sampling_model.fit(ballots, oracle_poll)

				for forecaster_seed in range(NUM_POLLS - 1):
					# get a seed
					simulated_ballots, simulated_counts = sampling_model.simulate_ballots(
						num_ballots=poll_size, 
						seed=forecaster_seed)
					# get a winner probability
					prob_actual_winner_wins = f(
						seed_counts=simulated_counts, 
						seed_ballots=simulated_ballots,
						n=np.sum(simulated_counts), 
						cands=cand_names,
						actual_winner=actual_winner)
					win_probabilities_across_seeds.append(prob_actual_winner_wins)
				forecaster_std_by_sampling_method[sampler][idx].append(np.std(win_probabilities_across_seeds))

		# print(oracle_probabilities)
		oracle_mean_by_sampling_rate.append(np.mean(oracle_probabilities))
		oracle_std_by_sampling_rate.append(np.std(oracle_probabilities))
		for forecaster in forecasters:
			forecaster_mean_by_sampling_rate[forecaster].append(
				np.mean(forecaster_probabilities[forecaster]))

	return sampling_rates, \
		oracle_mean_by_sampling_rate,\
		oracle_std_by_sampling_rate, \
		forecaster_mean_by_sampling_method, \
		forecaster_std_by_sampling_method

def process_one_election(election_tuple):
    """Worker function for a single election."""
    start = time.time()
    collection, election_name, ballots, ballot_counts, cand_names, skippped_votes = election_tuple
    fig_name = f"{collection}-{election_name[-6:-4]}"

    elim_votes = run_irv(len(cand_names), ballots.copy(), ballot_counts, cands=cand_names)
    actual_winner = max(elim_votes, key=elim_votes.get)

    sampling_rates, \
    oracle_means, \
    oracle_stds, \
    forecaster_stds = forecaster_vs_oracle(ballots, ballot_counts, cand_names, actual_winner)

    return (
        fig_name,
        sampling_rates,
        oracle_means,
        oracle_stds,
        forecaster_stds,
        time.time() - start
    )


	# file_name = "data/preflib/elections-all/burlington/ED-00005-00000002.toi" #burlington-election
	# fig_name = "burlington_2009"
	# ballots, ballot_counts, cand_names, skippped_votes = utils.read_preflib(file_name)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    # Ensure output directory exists
    out_dir = os.path.join("results", "preflib-resampling", args.output_dir)
    fig_dir = os.path.join("plots", args.output_dir)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)


    # Load elections
    elections = list(utils.load_all_preflib_elections(f'data/preflib/{args.data_dir}'))

    # Parallel computation
    num_workers = min(len(elections), cpu_count())
    print(f"Running with {num_workers} processes...")

    # Use imap_unordered to stream results as they complete
    with Pool(processes=num_workers) as pool:
        pbar = tqdm(total=len(elections), desc="Processing, saving, and plotting")
        for res in pool.imap_unordered(process_one_election, elections, chunksize=1):
            try:
                (
                    fig_name,
                    sampling_rates,
                    oracle_means,
                    oracle_stds,
                    forecaster_stds,
                    duration
                ) = res

                with open("timing.csv", "a") as f:
                	f.write(f"{fig_name},{duration}")

                # Save pickle immediately for this election
                out_path = os.path.join(out_dir, f"{fig_name}.pkl")
                with open(out_path, "wb") as f:
                    pickle.dump(
                        {
                            "fig_name": fig_name,
                            "sampling_rates": sampling_rates,
                            "oracle_means": oracle_means,
                            "oracle_stds": oracle_stds,
                            "forecaster_stds": forecaster_stds,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                print(f"Saved: {out_path}")

                # Plot right away (main process only)
                plot_forecaster_vs_oracle(
                    sampling_rates,
                    oracle_means,
                    oracle_stds,
                    [], # forecaster means
                    [("Bootstrap", forecaster_stds)],
                    fig_dir,
                    fig_name,
                )

                # Update progress
                pbar.update(1)
                pbar.set_postfix_str(fig_name)

            except Exception as e:
                # Keep going on individual failures
                print(f"Error processing an election: {e}")

