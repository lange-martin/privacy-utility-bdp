import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from plots_for_paper import MARKERS, COLORS, LINES

RNG = np.random.default_rng(seed=1253)


# Here follow all queries we test, deterministic and made DP or BDP using different bounds

def count_BB(datasets):
    return np.sum((datasets == 2), axis=1)


def count_active(datasets):
    return np.sum((datasets > 0), axis=1)


def sum(datasets):
    return np.sum(datasets, axis=1)


def laplace_mechanism(query, input_data, scale):
    noise = RNG.laplace(0, scale, len(input_data))
    return query(input_data) + noise, scale


def count_BB_bdp_general_bound(datasets, epsilon, **kwargs):
    m = kwargs['m']
    return laplace_mechanism(count_BB, datasets, m / epsilon)


def count_active_bdp_general_bound(datasets, epsilon, **kwargs):
    n = datasets.shape[1]
    return laplace_mechanism(count_active, datasets, n / epsilon)


def count_active_bdp_markov_chain_boun(datasets, epsilon, **kwargs):
    min_eps = 4 * np.log(np.max(kwargs['trans_probs'])/np.min(kwargs['trans_probs']))
    if epsilon == 1.:
        print(f"Minimum epsilon for Markov chain bound: {min_eps}")
    dp_eps = epsilon - min_eps
    if dp_eps > 0:
        return laplace_mechanism(count_active, datasets, 1.0 / dp_eps)
    else:
        raise ValueError


def count_active_dp(datasets, epsilon, **kwargs):
    return laplace_mechanism(count_active, datasets, 1.0 / epsilon)


def sum_bdp_general_bound(datasets, epsilon, **kwargs):
    m = datasets.shape[1]
    if 'm' in kwargs:
        m = kwargs['m']
    datasets = np.clip(datasets, kwargs['min_value'], kwargs['max_value'])
    return laplace_mechanism(sum, datasets, m * (kwargs['max_value'] - kwargs['min_value']) / epsilon)


def sum_bdp_gaussian_bound(datasets, epsilon, **kwargs):
    m = kwargs.get('m', datasets.shape[1])
    datasets = np.clip(datasets, kwargs['min_value'], kwargs['max_value'])
    bdp_factor = ((m**2)/(4*(1/kwargs['max_rho'] - m + 2)) + 1)*(kwargs['max_value'] - kwargs['min_value'])

    dp_eps = epsilon / bdp_factor
    return laplace_mechanism(sum, datasets, 1 / dp_eps)

def sum_dp(datasets, epsilon, **kwargs):
    datasets = np.clip(datasets, kwargs['min_value'], kwargs['max_value'])
    return laplace_mechanism(sum, datasets, (kwargs['max_value'] - kwargs['min_value']) / epsilon)


# Calculates the empirical transition probabilities of a binary Markov chain
def calc_transition_probabilities(dataset):
    inactive_to_inactive = 0
    inactive_to_active = 0
    active_to_inactive = 0
    active_to_active = 0

    prev_steps = 0
    current_steps = 0

    for index, row in dataset.iterrows():
        if index == 0:
            prev_steps = row['steps']
            continue

        current_steps = row['steps']
        if prev_steps > 0:
            if current_steps > 0:
                active_to_active += 1
            else:
                active_to_inactive += 1
        else:
            if current_steps > 0:
                inactive_to_active += 1
            else:
                inactive_to_inactive += 1

        prev_steps = current_steps

    times_inactive = inactive_to_inactive + inactive_to_active
    times_active = active_to_active + active_to_inactive

    print(
        f"Transition probabilities when inactive: ({inactive_to_inactive / times_inactive}, {inactive_to_active / times_inactive})")
    print(
        f"Transition probabilities when active: ({active_to_inactive / times_active}, {active_to_active / times_active})")

    return np.array([[inactive_to_inactive / times_inactive, inactive_to_active / times_inactive],
                     [active_to_inactive / times_active, active_to_active / times_active]])


# Function that executes the utility experiment for a specific deterministic query and DP/BDP queries ("private queries")
def run_experiment(datasets, non_private_query, private_queries, query_names, legend_loc='best', show_plot=True,
                   name=None, marker_start_index=0, min_y=None, max_y=None, min_y_mape=None, max_y_mape=None, show_legend=True, **kwargs):
    resolution = 20

    epsilons = np.linspace(1.0, 20.0, resolution)
    MAPE = np.empty((resolution, len(private_queries)))
    success = np.empty((resolution, len(private_queries))).astype(bool)
    real_results = non_private_query(datasets)
    print(f"Average result from deterministic query: {np.mean(real_results)}")

    abs_error_95 = np.empty((resolution, len(private_queries)))
    lp_scales = np.empty((resolution, len(private_queries)))

    for i, epsilon in tqdm(enumerate(epsilons), total=resolution):
        for k, private_query in enumerate(private_queries):
            try:
                private_results, lp_scales[i, k] = private_query(datasets, epsilon, **kwargs)
                error_results = private_results - real_results

                # Here we calculate the upper limit of the 95%-confidence interval of absolute errors
                abs_error_95[i, k] = np.quantile(np.abs(error_results), q=0.95)

                # Here we calculate the mean absolute percentage error
                MAPE[i, k] = np.mean(np.abs(np.divide(error_results, real_results)))
                success[i, k] = True
            except ValueError:
                success[i, k] = False

    # This is the confidence interval upper limit plot
    for i in range(len(private_queries)):
        plt.scatter(epsilons[success[:, i]], abs_error_95[success[:, i], i],
                     label=query_names[i], color=COLORS[i+marker_start_index], marker=MARKERS[i+marker_start_index], s=100)
        # Formula to calculate utility of the laplace mechanism based on the scale of the Laplacian distribution
        if any(lp_scale for lp_scale in lp_scales[success[:, i], i]):
            alphas = np.log(1 / 0.05) * lp_scales[success[:, i], i]
            plt.plot(epsilons[success[:, i]], alphas, color=COLORS[i+marker_start_index])

    if show_legend:
        plt.legend(loc=legend_loc, fontsize=20, handlelength=1., handletextpad=0.4)
    plt.xlabel('ε', fontsize=20)
    plt.yscale('log')
    plt.ylabel('α', fontsize=20, rotation='horizontal')
    plt.xticks(fontsize=20, ticks=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    plt.yticks(fontsize=20)
    plt.grid(visible=True, axis='both')
    plt.gca().set_axisbelow(True)
    if min_y is not None and max_y is not None:
        plt.ylim((min_y, max_y))
    if SAVE and name is not None:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{name}_confidence_interval.pdf', bbox_inches='tight')
    if show_plot:
        plt.show()

    # This is the MAPE plot
    for i in range(len(private_queries)):
        plt.scatter(epsilons[success[:, i]], MAPE[success[:, i], i],
                     label=query_names[i], color=COLORS[i+marker_start_index], marker=MARKERS[i+marker_start_index], s=100)

    if show_legend:
        plt.legend(loc=legend_loc, fontsize=20, handlelength=1., handletextpad=0.4)
    plt.xlabel('ε', fontsize=20)
    plt.yscale('log')
    plt.ylabel('MAPE', fontsize=20)
    plt.xticks(fontsize=20, ticks=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    plt.yticks(fontsize=20)
    plt.grid(visible=True, axis='both')
    plt.gca().set_axisbelow(True)
    if min_y_mape is not None and max_y_mape is not None:
        plt.ylim((min_y_mape, max_y_mape))
    if SAVE and name is not None:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{name}_MAPE.pdf', bbox_inches='tight')
    if show_plot:
        plt.show()


# Loads and discretizes the electricity dataset (in time and into different ranges/states)
def _load_and_discretize_electricity_data(file_path, time_period_hours):
    df = pd.read_csv(file_path)

    df['UNIX_TS'] = pd.to_datetime(df['UNIX_TS'], unit='s')
    df = df.set_index('UNIX_TS')

    # Dataset includes measurements of different sensors, we sum them all to get total consumption
    columns_to_sum = [col for col in df.columns]
    df['total_consumption_watt'] = df[columns_to_sum].sum(axis=1)

    # Resample to get average Watts over the new time_period_hours, then calculate Wh
    resampled_avg_watt = df['total_consumption_watt'].resample(f'{time_period_hours}H').mean().dropna()
    wh_summed_period = resampled_avg_watt * time_period_hours

    min_wh = wh_summed_period.min()
    max_wh = wh_summed_period.max()
    threshold = 80_000
    
    # Discretize into binary states using threshold
    discretized_states_series = (wh_summed_period > threshold).astype(int)
    discretized_states_series = discretized_states_series.dropna()
    
    prop_state_1 = discretized_states_series.mean()
    print(f"Range of electricity Wh: [{min_wh:.2f}, {max_wh:.2f}]")
    print(f"Threshold for discretization: {threshold:.2f}")
    print(f"Proportion of state 1: {prop_state_1:.2%}")

    return discretized_states_series


# Calculates the empirical transition probabilities between discretized states of electricity data
def _calculate_electricity_transition_matrix(discretized_states_series):
    transition_counts = np.zeros((2, 2), dtype=int)
    
    prev_state = -1

    for current_state in discretized_states_series:
        if prev_state != -1:
            transition_counts[prev_state, current_state] += 1
        prev_state = current_state
    
    # Convert counts to probabilities
    transition_probabilities = np.zeros((2, 2))
    for i in range(2):
        row_sum = np.sum(transition_counts[i, :])
        if row_sum > 0:
            transition_probabilities[i, :] = transition_counts[i, :] / row_sum

    return transition_probabilities


# Load data and execute experiment for Markov chain data (activity dataset)
def experiment_markov_activity():
    activity_data = pd.read_csv("datasets/activity.csv")
    activity_data = activity_data.dropna()

    probs = calc_transition_probabilities(activity_data)

    # All states in one dataset
    np_datasets = np.tile(activity_data['steps'].to_numpy(dtype=int), (1000, 1))

    run_experiment(np_datasets, 
                   count_active,
                   [count_active_bdp_general_bound, count_active_bdp_markov_chain_boun, count_active_dp],
                   ["", "", ""], 
                   show_legend=False, 
                   trans_probs=probs, 
                   min_y=1e-1, 
                   max_y=1e5,
                   name='markov_total'
                   )

    # Split dataframe into many dataframes, one per day
    data_per_day = [x for _, x in activity_data.groupby(['date'])]

    np_datasets = np.empty((len(data_per_day), len(data_per_day[0])))
    for i, df in enumerate(data_per_day):
        np_datasets[i] = df['steps'].to_numpy()

    run_experiment(np_datasets, 
                   count_active,
                   [count_active_bdp_general_bound, count_active_bdp_markov_chain_boun, count_active_dp],
                   ["General Bound", "Markov Chain Bound", "DP Query"], 
                   trans_probs=probs, 
                   min_y=1e-1,
                   max_y=1e5, 
                   name='markov_per_day', 
                   show_legend=False
                   )


# Load data and execute experiment for electricity data (Markov chain)
def experiment_markov_electricity(time_period_hours=24):
    file_path = "datasets/electricity_consumption_dataverse_files/Electricity_P.csv"
    discretized_states_series = _load_and_discretize_electricity_data(file_path, time_period_hours)

    trans_probs = _calculate_electricity_transition_matrix(discretized_states_series)
    print(f"Transition probabilities:\n{trans_probs}")

    print(f"Length of discretized_states_series: {len(discretized_states_series)}")

    # Prepare dataset: tile the single time series of states
    np_datasets = np.tile(discretized_states_series.to_numpy(dtype=int), (1000, 1))

    run_experiment(np_datasets, 
                   count_active,
                   [count_active_bdp_general_bound, count_active_bdp_markov_chain_boun, count_active_dp],
                   ["General Bound", "Markov Bound", "DP Query"],
                   show_legend=True,
                   legend_loc='right',
                   trans_probs=trans_probs,
                   m=np_datasets.shape[1],
                   name=f'markov_electricity_binary_{time_period_hours}h',
                   )


# Load data and execute experiment for Gaussian data
def experiment_gaussian_height():
    height_data = pd.read_stata("datasets/dataverse_files_galton_height/galton.dta")
    height_data = height_data.dropna()
    # Shuffle rows
    height_data = height_data.sample(frac=1, random_state=14)

    families_already_seen = set()
    # trio_datasets: Each row is a triple of child, mother, father
    trio_datasets = []
    # full_dataset: All heights, maximum 3 members per family.
    # Use set to remember to only use each family once.
    full_dataset = []
    for _, row in height_data.iterrows():
        trio_datasets.append([row['height'], row['father'], row['mother']])
        if row['family'] not in families_already_seen:
            full_dataset.append(row['height'])
            full_dataset.append(row['father'])
            full_dataset.append(row['mother'])
            families_already_seen.add(row['family'])
    trio_datasets = np.array(trio_datasets)
    full_dataset = np.array(full_dataset)
    print(f"Full dataset Galton length: {len(full_dataset)}")
    full_dataset = np.tile(full_dataset, (1000, 1))
    # Calculate maximum empirical Pearson correlation coefficient
    corr_matrix = np.corrcoef(trio_datasets.T)
    print("Pearson correlation coefficients")
    print(corr_matrix)
    max_rho = np.max([corr_matrix[0,1], corr_matrix[0,2], corr_matrix[1,2]])
    print(f"Maximum empirical Pearson correlation coefficient: {max_rho}")

    min_clip, max_clip = 0, 100
    print(f"Galton clipping range: {min_clip}, {max_clip}")

    run_experiment(full_dataset, 
                   sum, 
                   [sum_bdp_general_bound, sum_bdp_gaussian_bound, sum_dp],
                   ["General Bound/SOTA", "Gaussian Bound", "DP Query"], 
                   min_value=min_clip, 
                   max_value=max_clip, 
                   max_rho=max_rho,
                   m=3,
                   name='gaussian_galton_height'
                   )


def experiment_gaussian_iq():
    iq_data = pd.read_csv("datasets/kid_iq.csv")
    iq_data = iq_data[['kid_score', 'mom_iq']].dropna()

    kid_iqs = iq_data['kid_score']
    mom_iqs = iq_data['mom_iq']

    # Calculate Pearson correlation
    correlation_matrix = np.corrcoef(kid_iqs, mom_iqs)
    pearson_correlation = correlation_matrix[0, 1]
    print(f"Pearson correlation between kid IQ and mom IQ: {pearson_correlation:.4f}")
    
    max_rho = abs(pearson_correlation)

    # Create combined dataset
    combined_iqs = pd.concat([kid_iqs, mom_iqs]).to_numpy()
    
    min_iq = 0
    max_iq = 200
    print(f"IQ clipping range: min={min_iq}, max={max_iq}")

    # Tile the dataset 1000 times
    # Each row in np_datasets is a full copy of all combined_iqs
    np_datasets = np.tile(combined_iqs, (1000, 1))

    run_experiment(np_datasets, 
                   sum,
                   [sum_bdp_general_bound, sum_bdp_gaussian_bound, sum_dp],
                   ["General Bound/SOTA", "Gaussian Bound", "DP Query"],
                   min_value=min_iq, 
                   max_value=max_iq, 
                   max_rho=max_rho, 
                   m=2,  # number of correlated items
                   name='gaussian_iq',
                   legend_loc='upper right', 
                   show_plot=True
                   )


def experiment_gaussian_iq_synthetic():
    num_samples = 10000
    mean_val = 100
    std_dev = 15
    correlation = 0.45

    mean_vector = [mean_val, mean_val]
    variance = std_dev**2
    covariance = correlation * variance

    cov_matrix = np.array([
        [variance, covariance],
        [covariance, variance]
    ])

    # Generate the dataset of duos (child, mother)
    generated_duos_raw = RNG.multivariate_normal(mean_vector, cov_matrix, size=num_samples)

    # Flatten all scores into a single 1D array
    all_generated_scores_1d = generated_duos_raw.flatten()

    # Copy this 1D array 1000 times for the experiment
    duo_datasets = np.tile(all_generated_scores_1d, (1000, 1))

    # Define clipping range
    clip_min = 0
    clip_max = 200
    print(f"Synthetic IQ clipping range: min={clip_min}, max={clip_max}")

    run_experiment(duo_datasets, 
                   sum,
                   [sum_bdp_general_bound, sum_bdp_gaussian_bound, sum_dp],
                   ["General Bound/SOTA", "Gaussian Bound", "DP Query"],
                   min_value=clip_min,
                   max_value=clip_max,
                   max_rho=correlation,
                   m=2,
                   name='gaussian_iq_synthetic',
                   legend_loc='upper right',
                   show_plot=True
                   )


if __name__ == "__main__":
    SAVE = True
    experiment_markov_activity()
    experiment_gaussian_height()
    experiment_markov_electricity()
    experiment_gaussian_iq()
    experiment_gaussian_iq_synthetic()
