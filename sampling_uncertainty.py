import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 

from irv import run_irv
import utils

import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import pickle

import argparse

# Before: (NUM_POLLS^2 + NUM_POLLS) * NUM_FORECASTING_TRIALS = 5000 
# After: 2 * NUM_POLLS * NUM_FORECASTING_TRIALS = 3200
NUM_FORECASTING_TRIALS = 100
NUM_POLLS = 40
NUM_FORECASTER_STDS = 3


def f(seed_counts, seed_ballots, n, cands, actual_winner, n_trials=NUM_FORECASTING_TRIALS):
	'''
		Given a set of ballot counts from a poll and a known election size, calculate the probability that candidate "cand"
		wins across bootstrap trials, where in each trial the remaining ballots are sampled from seed_counts
	'''
	actual_winner_wins_counts = 0
	for seed in range(n_trials):
		simulated_election_counts = seed_counts
		if np.sum(seed_counts) < n: 
			additional_simulated_votes = utils.resample(seed_counts,
				sample_size=n-np.sum(seed_counts),
				with_replacement=True,
				seed=seed)
			simulated_election_counts = seed_counts + additional_simulated_votes
		assert np.sum(simulated_election_counts) == n

		elim_votes = run_irv(len(cands), seed_ballots.copy(), simulated_election_counts, cands=cands)
		simulated_winner = max(elim_votes, key=elim_votes.get)
		actual_winner_wins_counts += int(simulated_winner == actual_winner)
	return actual_winner_wins_counts / n_trials


def plot_forecaster_vs_oracle(sampling_rates,
	oracle_mean_by_sampling_rate,
	oracle_std_by_sampling_rate,
	forecaster_std_by_sampling_rate,
	fig_path,
	fig_name):
	
	fig, ax = plt.subplots(ncols=2, figsize=(10,5))

	ax[0].plot(sampling_rates, oracle_mean_by_sampling_rate, color="black", linestyle="--")

	ax[0].set_xlabel("Sampling Rate")
	ax[0].set_ylabel("Forecast Probability that Actual Winner Wins")
	ax[0].set_title(f"{fig_name} Forecast")


	ax[1].plot(sampling_rates, oracle_std_by_sampling_rate, color="black", linestyle="--")
	for idx, sampling_rate in enumerate(sampling_rates):
		ax[1].scatter(
			x=[sampling_rate] * len(forecaster_std_by_sampling_rate[idx]),
			y=forecaster_std_by_sampling_rate[idx],
			alpha=0.5,
			color="red")

	ax[1].set_xlabel("Sampling Rate")
	ax[1].set_ylabel("Std. Dev. Among Winner Forecasts")
	ax[1].set_title(f"{fig_name} Uncertainty")

	fig.savefig(f"{fig_path}/{fig_name}.pdf", bbox_inches="tight")


def forecaster_vs_oracle(ballots, ballot_counts, cand_names, actual_winner):
	# sampling_rates = np.arange(0.1, 1.05, 0.05)
	# sampling_rates = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
	sampling_rates = [0.01, 0.02, 0.04, 0.08]
	# sampling_rates = [0.1, 0.5, 1.0]

	oracle_mean_by_sampling_rate = [] #one float entry per sampling ratio
	forecaster_mean_by_sampling_rate = []

	oracle_std_by_sampling_rate = [] #one float entry per sampling ratio
	forecaster_std_by_sampling_rate = []

	for idx, sampling_rate in enumerate(sampling_rates):	
		oracle_probabilities = []
		forecaster_stds = []

		# ORACLE UNCERTAINTY
		for oracle_seed in range(NUM_POLLS):
			poll_size = max(int(np.sum(ballot_counts) * sampling_rate), 1)
			oracle_poll = utils.resample(ballot_counts, 
				sample_size=poll_size, 
				with_replacement=False,
				seed=oracle_seed+10)

			# Calculate the probability the actual winner wins 
			prob_actual_winner_wins = f(
				seed_counts=oracle_poll, 
				seed_ballots=ballots,
				n=np.sum(ballot_counts), 
				cands=cand_names,
				actual_winner=actual_winner)
			oracle_probabilities.append(prob_actual_winner_wins)
		
			if oracle_seed >= NUM_FORECASTER_STDS:
				continue

			# FORECASTER UNCERTAINTY
			win_probabilities_across_seeds = [prob_actual_winner_wins]
			for forecaster_seed in range(NUM_POLLS - 1):
				# get a seed
				simulated_poll = utils.resample(oracle_poll, 
					sample_size=poll_size, 
					with_replacement=True,
					seed=forecaster_seed)
				
				# get a winner probability
				prob_actual_winner_wins = f(
					seed_counts=simulated_poll, 
					seed_ballots=ballots,
					n=np.sum(ballot_counts), 
					cands=cand_names,
					actual_winner=actual_winner)
				win_probabilities_across_seeds.append(prob_actual_winner_wins)
			forecaster_stds.append(np.std(win_probabilities_across_seeds))

		# print(oracle_probabilities)
		oracle_mean_by_sampling_rate.append(np.mean(oracle_probabilities))
		oracle_std_by_sampling_rate.append(np.std(oracle_probabilities))

		forecaster_std_by_sampling_rate.append(forecaster_stds)
	return sampling_rates, \
		oracle_mean_by_sampling_rate,\
		oracle_std_by_sampling_rate, forecaster_std_by_sampling_rate

def process_one_election(election_tuple):
    """Worker function for a single election."""
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
                ) = res

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
                    forecaster_stds,
                    fig_dir,
                    fig_name,
                )

                # Update progress
                pbar.update(1)
                pbar.set_postfix_str(fig_name)

            except Exception as e:
                # Keep going on individual failures
                print(f"Error processing an election: {e}")

