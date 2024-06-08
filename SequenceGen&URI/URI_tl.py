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


vec_mean = np.vectorize(np.mean)


def firing_ranks(firing_times_per_trial):
    """
    Computes per firing rank the unit that had this firing rank from firing times per trial
    Input:  - firing_times_per_trial: 2d array containing for each trial, for each neuron the list of firing times
                                      that this neuron produced in this trial.
    Output: - firing_ranks:           Array containing for each rank in each trial which neuron had this rank in this trial
    """
    firing_ranks = np.zeros(firing_times_per_trial.shape) * np.nan
    # Loop over all trials
    for idx, trial in enumerate(firing_times_per_trial):
        # Take the average firing time
        mean_FT = vec_mean(trial)
        # Get the valid indices (not nan)
        ivalid = np.where(~np.isnan(mean_FT))[0]
        # Get the sort order (neuron per rank) on the valid indices
        isort = np.argsort(mean_FT[ivalid])
        # Get the original indices of all valid indices
        ranks = ivalid[isort]
        # Save the order
        firing_ranks[idx, :ranks.shape[0]] = ranks
    return firing_ranks


def firing_ranks2occ_matrix(firing_ranks):
    """
    Constructs an occurrence matrix from firing ranks where occ[x, y] represents how often neuron x had rank y.
    Input:  - firing_ranks: Array containing for each rank in each trial which neuron had this rank in this trial
    Output: - occ:          Occurrence matrix
    """
    n_neurons = firing_ranks.shape[1]
    occ = np.zeros((n_neurons, n_neurons), dtype=int)
    # Loop over trials
    for trial in firing_ranks:
        # Loop over ranks:
        for rank, neuron in enumerate(trial):
            if np.isnan(neuron):
                continue
            # For not nan rank, neuron pairs, increment the respective square in the occurrence matrix|
            occ[int(neuron), rank] += 1
    return occ


def get_surrogate_MIs(firing_ranks, n_surrogates):
    """
    Computes a number of MI scores belonging to unit-rank shuffled versions of firing_ranks
    Input:  - firing_ranks: Array containing for each rank in each trial which neuron had this rank in this trial
            - n_surrogates: Number of shuffles / number of surrogate scores obtained
    Output: - MIs:          MI scores belonging to the surrogates (1 score per surrogate)
    """
    # Create a copy of firing_ranks
    ranks = firing_ranks.copy()
    # Create an array for return values
    MIs = np.empty(n_surrogates)
    # Loop n_surrogates times
    for nr in range(n_surrogates):
        for trial in ranks:
            # Get the valid indices (not nan)
            ivalid = np.where(~np.isnan(trial))[0]
            # Shuffle trials and ranks
            trial[ivalid] = trial[rng.permutation(ivalid)]
        occ = firing_ranks2occ_matrix(ranks)
        MIs[nr] = MI(occ)
    return MIs


def getURI(firing_times_per_trial, nr_surrogates):
    """
    Computes the mutual information between unit and firing rank.
    Firing rank is defined as the place in the order of average firing times for a neuron within a trial.
    Input:  - firing_times_per_trial: 2d array containing for each trial, for each neuron the list of firing times
                                      that this neuron produced in this trial.
            - nr_surrogates:          Number of shuffled occurrence matrices to compute p-value. Gets more accurate
                                      with higher numbers.
    Output: - URI:                    The calculated URI metric.
            - occurrence_matrix:      The occurrence matrix between neuron and rank
            - p-value:                p-value from surrogate calculation. Can be interpreted as the probability of
                                      URI metric being as high as it is under null-hypothesis (no sequence).
    """
    ranks = firing_ranks(firing_times_per_trial)
    occurrence_matrix = firing_ranks2occ_matrix(ranks)
    URI = MI(occurrence_matrix)
    surrogates = get_surrogate_MIs(ranks, nr_surrogates)
    URI_p_val = np.count_nonzero(surrogates >= URI) / nr_surrogates
    avg_surr = np.mean(surrogates)
    surr_95 = np.sort(surrogates)[int(nr_surrogates*0.95)]
    return URI, URI_p_val, occurrence_matrix, avg_surr, surr_95