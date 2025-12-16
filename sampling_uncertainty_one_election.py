import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from irv import run_irv
import utils
from model import Bootstrap, PLModel, ContextModel

import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import pickle
import argparse
import time
from collections import defaultdict


# Before: (NUM_POLLS^2 + NUM_POLLS) * NUM_FORECASTING_TRIALS = 5000
# After: 2 * NUM_POLLS * NUM_FORECASTING_TRIALS = 3200
NUM_FORECASTING_TRIALS = 50
NUM_POLLS = 100

def get_fig_name(election_tuple):
    collection, election_name, _, _, _, _ = election_tuple
    return f"{collection}-{election_name[-6:-4]}"


def read_group_config(config_file: str):
    election_to_group = {}
    with open(f"config/{config_file}", "r") as f:
        for line in f.readlines():
            line_entries = line.split(",")
            assert len(line_entries) == 2
            election_to_group[line_entries[0]] = int(line_entries[1])
    return election_to_group


def f(
    seed_counts,
    seed_ballots,
    n_total,
    n_simulated,
    cands,
    actual_winner,
    n_trials: int = NUM_FORECASTING_TRIALS,
    sampler: str = "Bootstrap",
):
    """
    Given a set of ballot counts from a poll and a known election size, calculate
    the probability that `actual_winner` wins across bootstrap-style trials.
    In each trial, the remaining ballots are sampled from (seed_ballots, seed_counts)
    using the specified sampler.
    """
    actual_winner_wins_counts = 0

    sampling_model = utils.get_model_object(sampler, cands)

    sampling_model.fit(seed_ballots, seed_counts)

    for seed in range(n_trials):
        simulated_ballots, simulated_counts = sampling_model.simulate_ballots(
            num_ballots=n_simulated,
            seed=seed,
        )

        simulated_ballots.extend(seed_ballots.copy())
        simulated_counts = simulated_counts + tuple(seed_counts)

        # assert np.sum(simulated_counts) == n_total

        elim_votes = run_irv(len(cands), simulated_ballots.copy(), simulated_counts, cands=cands)
        simulated_winner = max(elim_votes, key=elim_votes.get)
        actual_winner_wins_counts += int(simulated_winner == actual_winner)

    return actual_winner_wins_counts / n_trials


def _forecast_task(args):
    """
    Worker used inside `forecaster_vs_oracle`.

    Each task corresponds to a single (sampling_rate, oracle_seed, forecaster)
    triple for one fixed election.
    """
    (
        sampling_rate,
        oracle_seed,
        forecaster,
        ballots,
        ballot_counts,
        cand_names,
        actual_winner,
    ) = args

    total_n = int(np.sum(ballot_counts))
    poll_size = max(int(total_n * sampling_rate), 1)

    # Draw a poll from the full election
    oracle_poll = utils.resample(
        ballot_counts,
        sample_size=poll_size,
        with_replacement=False,
        seed=oracle_seed + 10,
    )

    # Compute probability that actual_winner ultimately wins using this forecaster

    if forecaster == "Oracle":
        prob_actual_winner_wins = f(
            seed_counts=ballot_counts,
            seed_ballots=ballots,
            n_total=total_n,
            n_simulated=total_n - poll_size,
            cands=cand_names,
            actual_winner=actual_winner,
            sampler="Bootstrap",
        )
    else:
        prob_actual_winner_wins = f(
            seed_counts=oracle_poll,
            seed_ballots=ballots,
            n_total=total_n - poll_size,
            n_simulated=poll_size,
            cands=cand_names,
            actual_winner=actual_winner,
            sampler=forecaster,
        )

    return sampling_rate, oracle_seed, forecaster, prob_actual_winner_wins


def plot_forecaster_vs_oracle(
    sampling_rates,
    oracle_mean_by_sampling_rate,
    oracle_std_by_sampling_rate,
    forecaster_mean_by_sampling_method,
    forecaster_std_by_sampling_method,
    fig_path,
    fig_name,
):
    """
    Plot mean forecast probability (left) and standard deviation over polls (right).
    Oracle (Bootstrap) is plotted in black; forecasters use colors from utils.colors.
    """
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    # MEANS
    ax[0].plot(
        sampling_rates,
        oracle_mean_by_sampling_rate,
        color="black",
        linestyle="--",
        label="Bootstrap (oracle)",
    )

    for forecaster_idx, (forecaster_name, forecaster_means) in enumerate(
        forecaster_mean_by_sampling_method.items()
    ):
        # Optionally skip double-plotting the oracle if it's in the dict
        if forecaster_name == "Bootstrap":
            continue

        ax[0].plot(
            sampling_rates,
            forecaster_means,
            linestyle="--",
            color=utils.colors[forecaster_idx % len(utils.colors)],
            label=forecaster_name,
        )

    ax[0].set_xlabel("Sampling Rate")
    ax[0].set_ylabel("Forecast Probability that Actual Winner Wins")
    ax[0].set_title(f"{fig_name} Forecast")
    ax[0].legend()

    # STD DEV
    ax[1].plot(
        sampling_rates,
        oracle_std_by_sampling_rate,
        color="black",
        linestyle="--",
        label="Bootstrap (oracle)",
    )

    for forecaster_idx, (forecaster_name, forecaster_stds) in enumerate(
        forecaster_std_by_sampling_method.items()
    ):
        if forecaster_name == "Bootstrap":
            continue

        ax[1].plot(
            sampling_rates,
            forecaster_stds,
            marker="o",
            linestyle="--",
            color=utils.colors[forecaster_idx % len(utils.colors)],
            label=forecaster_name,
        )

    ax[1].set_xlabel("Sampling Rate")
    ax[1].set_ylabel("Std. Dev. Across Polls")
    ax[1].set_title(f"{fig_name} Uncertainty")
    ax[1].legend()

    os.makedirs(fig_path, exist_ok=True)
    fig.savefig(os.path.join(fig_path, f"{fig_name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def forecaster_vs_oracle(
    ballots,
    ballot_counts,
    cand_names,
    actual_winner,
):
    """
    For a single election, compare an oracle forecaster and several model-based
    forecasters across a grid of sampling rates.

    Parallelization is done *within* this function: each (sampling_rate,
    oracle_seed, forecaster) triple is dispatched to one CPU.
    """
    sampling_rates = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]

    # Treat the oracle ("Bootstrap") as just another forecaster
    forecasters = [
        "Bootstrap",
        # "Oracle", 
        # "Contextual By Length", 
        # "Mallows Dispersion 0.5",
        # "Mallows Dispersion 1",
        # "Mallows Dispersion 5",
        # "Contextual Perturbation Dispersion 0.5",
        # "Contextual Perturbation Dispersion 1",
        # "Contextual Perturbation Dispersion 5",
        # "Uniform"
        "Contextual Truncated",
        "Contextual Truncated Weighted First",
        "Contextual Truncated Weighted Mtx",
        "Contextual Truncated Weighted First and Mtx"
    ]
    oracle_name = "Bootstrap"

    tasks = []
    for sampling_rate in sampling_rates:
        for oracle_seed in range(NUM_POLLS):
            for forecaster in forecasters:
                tasks.append(
                    (
                        sampling_rate,
                        oracle_seed,
                        forecaster,
                        ballots,
                        ballot_counts,
                        cand_names,
                        actual_winner,
                    )
                )

    num_workers = cpu_count()
    num_workers = max(1, min(num_workers, len(tasks)))

    if num_workers == 1:
        results = [_forecast_task(task) for task in tasks]
    else:
        with Pool(processes=num_workers) as pool:
            results = []
            for res in tqdm(
                pool.imap_unordered(_forecast_task, tasks, chunksize=1), 
                total=len(tasks)
            ):
                results.append(res)

    # Aggregate probabilities by (forecaster, sampling_rate)
    probs = defaultdict(list)  # key: (forecaster, sampling_rate) -> list[prob]
    for sampling_rate, oracle_seed, forecaster, prob in results:
        probs[(forecaster, sampling_rate)].append(prob)

    oracle_mean_by_sampling_rate = []
    oracle_std_by_sampling_rate = []
    forecaster_mean_by_sampling_method = {f: [] for f in forecasters}
    forecaster_std_by_sampling_method = {f: [] for f in forecasters}

    for sampling_rate in sampling_rates:
        oracle_probs = probs[(oracle_name, sampling_rate)]
        print(oracle_probs)
        oracle_mean_by_sampling_rate.append(np.mean(oracle_probs))
        oracle_std_by_sampling_rate.append(np.std(oracle_probs))

        for forecaster in forecasters:
            fp = probs[(forecaster, sampling_rate)]
            forecaster_mean_by_sampling_method[forecaster].append(np.mean(fp))
            forecaster_std_by_sampling_method[forecaster].append(np.std(fp))

    return (
        sampling_rates,
        oracle_mean_by_sampling_rate,
        oracle_std_by_sampling_rate,
        forecaster_mean_by_sampling_method,
        forecaster_std_by_sampling_method,
    )


def process_one_election(election_tuple):
    """
    Helper for processing a single election tuple of the form returned by
    `utils.load_all_preflib_elections`.
    """
    start = time.time()

    fig_name = get_fig_name(election_tuple)
    print(f"Processing: {fig_name}")

    collection, election_name, ballots, ballot_counts, cand_names, skippped_votes = election_tuple

    elim_votes = run_irv(len(cand_names), ballots.copy(), ballot_counts, cands=cand_names)
    actual_winner = max(elim_votes, key=elim_votes.get)

    (
        sampling_rates,
        oracle_means,
        oracle_stds,
        forecaster_means,
        forecaster_stds,
    ) = forecaster_vs_oracle(
        ballots,
        ballot_counts,
        cand_names,
        actual_winner,
    )

    duration = time.time() - start

    return (
        fig_name,
        sampling_rates,
        oracle_means,
        oracle_stds,
        forecaster_means,
        forecaster_stds,
        duration,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--fig_name", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    # Ensure output directories exist
    out_dir = os.path.join("results", "preflib-resampling", args.output_dir)
    fig_dir = os.path.join("plots", args.output_dir)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load elections
    elections = list(utils.load_all_preflib_elections(f"data/preflib/{args.data_dir}"))

    # Configure global macros
    print(f"NUM_FORECASTING_TRIALS: {NUM_FORECASTING_TRIALS}")
    print(f"NUM_POLLS: {NUM_POLLS}")

    # Sequential over elections; parallelization is inside forecaster_vs_oracle
    for election in elections:
        if get_fig_name(election) != args.fig_name:
            continue 

        (
            fig_name,
            sampling_rates,
            oracle_means,
            oracle_stds,
            forecaster_means,
            forecaster_stds,
            duration,
        ) = process_one_election(election)

        # Save pickle immediately for this election
        out_path = os.path.join(out_dir, f"{fig_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "fig_name": fig_name,
                    "sampling_rates": sampling_rates,
                    "oracle_means": oracle_means,
                    "oracle_stds": oracle_stds,
                    "forecaster_means": forecaster_means,
                    "forecaster_stds": forecaster_stds,
                },
                f,
            )

        # Plot right away
        plot_forecaster_vs_oracle(
            sampling_rates,
            oracle_means,
            oracle_stds,
            forecaster_means,
            forecaster_stds,
            fig_dir,
            fig_name,
        )


