import numpy as np
rng = np.random.default_rng(42)

def MI(M):
    """
    Calculates the mutual information between the 2 axes of a matrix.
    Input:  - M:  2d matrix
    Output: - MI: Mutual information
    """
    sizex = M.shape[0]
    sizey = M.shape[1]
    total = np.sum(M)
    p_Y = np.sum(M, axis=0) / total
    p_X = np.sum(M, axis=1) / total
    # MI = sum over x and y: p(x,y) * log(p(x,y) / (p(x) * p(y)))
    #    = sum over x and y: M(x,y)/total * log((M(x,y) * total) / (p_X(x) * p_Y(y)))
    MI = sum([sum([(
        0 if M[x, y] == 0 else
        M[x, y] / total * np.log((M[x, y] / total) / (p_X[x] * p_Y[y]))
    ) for x in range(sizex)]) for y in range(sizey)])
    return MI

def spikes_in_window(spikes, window_start, window_end):
    return [spike for spike in spikes if spike > window_start and spike < window_end]

cut_spikes = np.vectorize(spikes_in_window, otypes=[object])
vec_mean = np.vectorize(np.mean)
len_vec = np.vectorize(len)

def align_with_events(trials, neuron, window=(-0.1, 0.1)):
    """
    Procedure to obtain event-aligned trials. This procedure creates a window around each spike of the specified neuron
    and keeps it if the firing neuron is also the one that has a mean firing time closest to 0 (relative to the window)
    out of all neurons.
    Input:  - trials:         Initial trials, not aligned to spikes.
            - neuron:         Index of the neuron whose spikes are markers.
            - window:         Size of the window, tuple containing start and end relative to the spike in seconds.
    Output: - aligned_trials: Array containing all the event trials aligned to spikes of the specified neuron.
    """
    idx = 0
    max_nr_trials = np.sum(len_vec(trials[:, neuron]))
    aligned_trials = np.empty((max_nr_trials, trials.shape[1]), dtype=object)
    # Loop over trials
    for trial in trials:
        # Get the spikes
        spikes = trial[neuron]
        # Loop over spikes
        for spike in spikes:
            # Determine the window
            window_start = spike + window[0]
            window_end = spike + window[1]
            # Cut out all spikes in the trial within this window
            new_trial = cut_spikes(trial, window_start, window_end)
            # Get mean firing time per neuron
            MFT = vec_mean(new_trial)
            # Keep this trial if the specified neuron has MFT closest to the spike time
            aligned_trials[idx, :] = new_trial
            idx += 1
    return aligned_trials[:idx, :]

def aligned_firing_ranks(firing_times_per_trial, aligned_neuron):
    """
    Computes per firing rank the unit that had this relative firing rank to the aligned neuron from firing times per trial.
    To obtain the index for the kth relative rank, look at index (k + #neurons - 1).
    Input:  - firing_times_per_trial: 2d array containing for each trial, for each neuron the list of firing times
                                      that this neuron produced in this trial.
            - aligned_neuron:         index of the neuron that the trials are aligned to.
    Output: - firing_ranks:           Array containing for each rank in each trial which neuron had this relative rank in this trial
    """
    firing_ranks = np.zeros((firing_times_per_trial.shape[0], firing_times_per_trial.shape[1]*2-1))*np.nan
    # Loop over all trials
    for idx, trial in enumerate(firing_times_per_trial):
        # Take the average firing time
        mean_FT = vec_mean(trial)
        # If the aligned neuron did not have a firing time, exclude the trial
        # (this should not occur if the trials are aligned to this neuron)
        if np.isnan(mean_FT[aligned_neuron]):
            continue
        # Get the valid indices (not nan)
        ivalid = np.where(~np.isnan(mean_FT))[0]
        # Get the sort order (neuron per rank) on the valid indices
        isort = np.argsort(mean_FT[ivalid])
        # Get the original indices of all valid indices
        ranks = ivalid[isort]
        # Get firing rank of aligned neuron
        rank_of_aligned = np.where(ranks == aligned_neuron)[0][0]
        # Save the adjusted order (padded by nan)
        firing_ranks[idx, firing_times_per_trial.shape[1] - rank_of_aligned - 1: firing_times_per_trial.shape[1] - rank_of_aligned + ranks.shape[0] - 1]\
        = ranks
    return firing_ranks

def aligned_firing_ranks2occ_matrix(firing_ranks):
    """
    Constructs an occurrence matrix from firing ranks where occ[x, y] represents how often neuron x had rank y.
    Intended for relative ranks use case.
    Input:  - firing_ranks: Array containing for each rank in each trial which neuron had this rank in this trial
    Output: - occ:          Occurrence matrix
    """
    n_neurons = int((firing_ranks.shape[1]+1)/2)
    occ = np.zeros((n_neurons, n_neurons*2-2), dtype=int)
    # Loop over trials
    for trial in firing_ranks:
        # Loop over ranks:
        for rank, neuron in enumerate(trial):
            if np.isnan(neuron) or rank == n_neurons-1:
                continue
            # For not nan rank, neuron pairs, increment the respective square in the occurrence matrix
            if rank > n_neurons-1:
                occ[int(neuron), rank-1] += 1
            else:
                occ[int(neuron), rank] += 1
    return occ

def surrogate_occ(firing_times_per_trial, window=(-0.1, 0.1)):
    """
    Get the occurrence matrix between neurons and ranks for a randomly aligned trial.
    input:  - firing_times_per_trial: 2d array containing for each trial, for each neuron the list of firing times
                                      that this neuron produced in this trial.
            - window:                 tuple containing the start and end of the window set around each spike.
    output: - occ:                    The occurrence matrix of the trial
    """
    while True:
        trial = rng.integers(0, firing_times_per_trial.shape[0])
        min_time = np.min([np.min(firing_times_per_trial[trial, neuron]) for neuron in range(firing_times_per_trial.shape[1])])
        max_time = np.max([np.max(firing_times_per_trial[trial, neuron]) for neuron in range(firing_times_per_trial.shape[1])])
        alignment = rng.uniform(min_time, max_time)
        relevant_spikes = np.array([cut_spikes(firing_times_per_trial[trial], alignment + window[0], alignment + window[1])])
        spiking_neurons = [idx for idx, neuron_spikes in enumerate(relevant_spikes[0]) if len(neuron_spikes) > 0]
        if len(spiking_neurons) == 0:
            continue
        align_to = spiking_neurons[rng.integers(0, len(spiking_neurons))]
        ranks = aligned_firing_ranks(relevant_spikes, align_to)
        return aligned_firing_ranks2occ_matrix(ranks)

def surrogate_occ_alt(firing_times_per_trial, window=(-0.1, 0.1)):
    # Pick random trial
    trial = rng.integers(0, firing_times_per_trial.shape[0])
    # Pick random neuron
    align_to = rng.integers(0, firing_times_per_trial.shape[1])
    # Pick random spike
    spike_time = rng.choice(firing_times_per_trial[trial, align_to])
    relevant_spikes = np.array([cut_spikes(firing_times_per_trial[trial], spike_time + window[0], spike_time + window[1])])
    ranks = aligned_firing_ranks(relevant_spikes, align_to)
    return aligned_firing_ranks2occ_matrix(ranks)

def occ_random_spike(firing_times_per_trial, window=(-0.1, 0.1)):
    get_nr_spikes = np.vectorize(len)
    nr_spikes = get_nr_spikes(firing_times_per_trial)
    p = nr_spikes.flatten() / np.sum(nr_spikes)
    # Try up to 100 times to find a spike
    for i in range(100):
        idx = rng.choice(np.arange(nr_spikes.flatten().shape[0]), p=p)
        trial = int(idx / firing_times_per_trial.shape[1])
        neuron = idx % firing_times_per_trial.shape[1]
        if not firing_times_per_trial[trial, neuron]:
            print("index "+ str(idx))
            continue
        spike_time = rng.choice(firing_times_per_trial[trial, neuron])
        relevant_spikes = np.array([cut_spikes(firing_times_per_trial[trial], spike_time + window[0], spike_time + window[1])])
        ranks = aligned_firing_ranks(relevant_spikes, neuron)
        return aligned_firing_ranks2occ_matrix(ranks)
    # Else give an empty occ matrix
    print("WARNING: could not find spike")
    return np.zeros((firing_times_per_trial.shape[1], firing_times_per_trial.shape[1]))

def get_URI_nontimelocked(firing_times_per_trial, nr_surrogates, window=(-0.1, 0.1)):
    """
    Computes the mutual information between unit and firing rank after aligning windows at spikes of every neuron, per neuron.
    Firing rank is defined as the place in the order of average firing times for a neuron within a trial.
    Input:  - firing_times_per_trial: 2d array containing for each trial, for each neuron the list of firing times
                                      that this neuron produced in this trial.
            - nr_surrogates:          Number of shuffled occurrence matrices to compute p-value. Gets more accurate
                                      with higher numbers.
            - window:                 tuple containing the start and end of the window set around each spike.
    Output: - URI:                    The calculated URI metric.
            - occurrence_matrix:      The occurrence matrix between neuron and rank
            - p-value:                p-value from surrogate calculation. Can be interpreted as the probability of
                                      URI metric being as high as it is under null-hypothesis (no sequence).
    """
    n_neurons = firing_times_per_trial.shape[1]
    URIs = np.zeros(n_neurons)
    nr_windows = np.zeros(n_neurons, dtype=int)
    for neuron in range(n_neurons):
        aligned_trials = align_with_events(firing_times_per_trial, neuron, window=window)
        nr_windows[neuron] = len(aligned_trials)
        firing_ranks = aligned_firing_ranks(aligned_trials, neuron)
        occ_matrix = aligned_firing_ranks2occ_matrix(firing_ranks)
        URI_val = MI(occ_matrix)
        URIs[neuron] = URI_val
    max_nr_windows = np.max(nr_windows)
    surrogate_stronger = np.zeros(n_neurons, dtype=int)
    for surrogate in range(nr_surrogates):
        surr_occ = [occ_random_spike(firing_times_per_trial, window=window) for _ in range(int(max_nr_windows))]
        for neuron in range(n_neurons):
            surr_URI = MI(np.sum(surr_occ[:nr_windows[neuron]], axis=0))
            if surr_URI >= URIs[neuron]:
                surrogate_stronger[neuron] += 1
    URI_p_vals = surrogate_stronger / nr_surrogates
    return URIs, URI_p_vals