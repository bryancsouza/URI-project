import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)

def poisson_FR(n_neurons: int, FR, duration: float):
    """
    Generates neuronal activity from poisson firing rate for one neurons.
    Input:  - n_neurons:   The number of neurons to simulate.
            - FR:          The firing rate of the poisson process. (array<float>)
            - duration:    The length of recording being simulated in seconds.
    Output: - num_spikes:  The total number of spikes generated.
            - spike_times: Spike timings, sorted.
            - ISI:         Inter-spike intervals between all spikes.
    """
    # Generate number of spikes from poisson distribution
    num_spikes = rng.poisson(FR * duration)
    # Initialize arrays for spike timing and inter-spike interval
    spike_times = np.empty(n_neurons, dtype=object)
    ISI = np.empty(n_neurons, dtype=object)
    for n in range(n_neurons):
        # Draw spike timings from uniform distribution
        spike_times[n] = list(np.sort(rng.uniform(0, duration, num_spikes[n])))
        # Calculate ISI per neuron
        ISI[n] = np.diff(spike_times[n])
    return num_spikes, spike_times, ISI

def random_FR(n_neurons, mean, sigm):
    """
    Draws an array of random firing rates from a gaussian distribution for a specified amount of neurons.
    input:  - n_neurons: The amount of neurons to draw a firing rate for.
            - mean:      The mean firing rate.
            - sigm:      Sigma (spread) of the firing rates.
    output: - FRs:       Array of firing rates (one FR for each neuron).
    """
    return rng.normal(mean, sigm, n_neurons)


def random_sequence_template(n_neurons, t_start, t_end, nr_in_sequence):  # Experiment with time of sequence
    """
    Generate a template from which sequences can be generated. A template is defined as a set of neuron / spike time
    pairs which represent the mean of where and when spikes are expected to happen.
    input:  - nr_neurons:     The total number of neurons in the virtual recording.
            - t_start:        The earliest point in time at which a sequence can begin in seconds.
            - t_end:          The latest point in time at which a sequence can end in seconds.
            - nr_in_sequence: The number of neurons participating in the sequence.
    output: - timings:        The times that neurons are expected to fire

    """
    # Sample which neurons participate in the sequence
    participating = rng.integers(0, n_neurons, nr_in_sequence)
    # Initialize array for keeping track of timings.
    timings = np.empty(n_neurons, dtype=object)
    # Set each element to an empty array
    for i in range(n_neurons):
        timings[i] = []
    # For every neuron participating, instead generate a random timing
    for n in participating:
        timings[n].append(rng.uniform(t_start, t_end))
    return timings, participating

def template2sequence(timings, jitter, delay_var=0, mean_delay=0):
    """
    Generates a sequence from a sequence template. The resulting sequence resembles the template with some variations
    in spike timing.
    Input:  - timings:       The spike timings in the template.
            - jitter:        Standard deviation of spike timing.
    Output: - spike_timings: The spike timings in the resulting sequence.
    """
    spike_timings = np.empty(timings.shape[0], dtype=object)
    for i in range(timings.shape[0]):
        delay = rng.normal(mean_delay, delay_var)
        spike_timings[i] = []
        for t in timings[i]:
            t_spike = rng.normal(t, jitter)
            spike_timings[i].append(t_spike + delay)
    return spike_timings

def embed_sequence(noise, sequence, mode="remove_random"):
    """
    Adds activity and sequence together, removing one spike from the random activity for every spike in the sequence.
    Input:  - noise:         Random poisson firing to embed the sequence in.
            - sequence:      The sequence being embeded
            - mode:          If mode="remove_nearest", nearest spike to sequence spikes will be removed. (currently unimplemented)
                             Add mode: If you add a spike, remove one and increase a similar ISI elsewhere.
                             If you add a spike, you shorten the ISI, so you can go somewhere else with the same ISI and lengthen that ISI.
                             If mode="remove_random", a random spike will be removed from the spike train.
    Output: - spike_timings: Spike timings after embedding
    """
    spike_timings = np.empty(noise.shape[0], dtype=object)
    for i in range(noise.shape[0]):
        if mode == "remove_random":
            spike_timings[i] = list(rng.choice(noise[i], max(len(noise[i]) - len(sequence[i]), 0), replace=False))
        elif mode == "add":
            spike_timings[i] = noise[i]
        else:
            raise Exception("Replacement methods other than \"remove_random\" are not implemented yet.")
        spike_timings[i].extend(sequence[i])
        spike_timings[i].sort()
    return spike_timings

def simulate_trials(n_trials, n_neurons, duration, seq_len, mean_delay=0, delay_var=0,
                    avg_FR=5, FR_var=0.2,
                    has_sequence=False, jitter=0.001, nr_in_sequence=5):
    """
    Simulates a number of trials with a common underlying sequence.
    Firing rate varies between neurons, but is constant per neuron across trials.
    Input:  - n_trials:       Number of trials simulated.
            - n_neurons:      Number of neurons per trial.
            - duration:       Duration of a trial.
            - avg_FR:         Average background firing rate across neurons.
            - FR_var:         Variation of firing rate between neurons.
            - has_sequence:   If True, embeds a common sequence into the activity.
            - jitter:         Jitter / time variability of spikes within the sequence.
            - nr_in_sequence: Number of neurons (spikes) participating in the sequence.
    Output: - trials:         Array containing for each trial, for each neuron the spike times.
    """
    # Generate common sequence
    template, neurons = random_sequence_template(n_neurons, 0, seq_len, nr_in_sequence)
    # Determine FR per neuron
    FR = random_FR(n_neurons, avg_FR, FR_var)
    # Make array to store trials
    trials = np.empty((n_trials, n_neurons), dtype=object)
    # Keep track of sequence timings
    sequences = np.empty((n_trials, n_neurons), dtype=object)
    for i in range(n_trials):
        # Generate background activity
        num_spikes, noise, ISI = poisson_FR(n_neurons, FR, duration)
        if has_sequence:
            # Generate sequence
            sequence = template2sequence(template, jitter, delay_var, mean_delay)
            sequences[i, :] = sequence
            # Embed suence into background activity
            trials[i] = embed_sequence(noise, sequence)
        else:
            trials[i] = noise
    return trials, sequences

def simulate_unaligned(n_trials, n_neurons, duration, avg_FR=5, FR_var=0.2,
                       time_btwn_seq=4, seq_length=0.05, jitter=0.001, nr_in_sequence=5):
    """
    Simulates a number of trials with a commom underlying sequence that repeats within each trial
    as well as between trials, but not at the same time.
    Input:  - n_trials:       Number of trials simulated.
            - n_neurons:      Number of neurons per trial.
            - duration:       Duration of a trial.
            - avg_FR:         Average background firing rate across neurons.
            - FR_var:         Variation of firing rate between neurons.
            - time_btwn_seq:  The average time between sequences.
            - seq_length:     The length of a sequence.
            - jitter:         Jitter / time variability of spikes within the sequence.
            - nr_in_sequence: Number of neurons (spikes) participating in the sequence.
    Output: - trials:         Array containing for each trial, for each neuron the spike times.
    """
    # Generate common sequence
    template, participating = random_sequence_template(n_neurons, 0, seq_length, nr_in_sequence)
    # Determine FR per neuron
    FR = random_FR(n_neurons, avg_FR, FR_var)
    # Make array to store trials
    trials = np.empty((n_trials, n_neurons), dtype=object)
    for i in range(n_trials):
        # Generate background activity
        num_spikes, noise, ISI = poisson_FR(n_neurons, FR, duration)
        # Get the sequence markers
        delay = 0
        # Generate the first sequence (this might break if the delay is randomly too long)
        delay += rng.exponential(scale=time_btwn_seq)
        sequence = template2sequence(template, jitter, mean_delay=delay)
        delay += seq_length
        # Keep generating sequences until there is no space left
        while duration - delay > seq_length:
            # Delay by a random amount
            delay += rng.exponential(scale=time_btwn_seq)
            # Create a sequence and embed it
            seq = template2sequence(template, jitter, mean_delay=delay, delay_var=0)
            sequence = embed_sequence(sequence, seq, mode="add")
            # Delay by the length of the sequence
            delay += seq_length
        trials[i] = embed_sequence(noise, sequence)
    return trials, sequence, participating

def plot_activity(n_neurons, spike_times, duration, title, sequence=None, color_spike='k', color_seq='r', figsize=[14, 8]):
    """
    Plots spike trains across multiple neurons over time.
    Input: - n_neurons:   The number of neurons to be plotted.
           - spike_times: The timings of spikes that will be plotted. (array<list<float>>)
           - duration:    The time window that the computation takes place in.
           - title:       Common title over the whole figure
    """
    # Create subplots with given title
    fig, ax = plt.subplots(n_neurons, sharex=False, figsize=figsize)
    fig.suptitle(title)
    # Remove padding
    plt.subplots_adjust(wspace=0, hspace=0)
    # For each neuron and each spike time for this neuron, plot a vertical line for a spike
    for nr, n in enumerate(ax):
        for t in spike_times[nr]:
            n.axvline(t, color=color_spike)
        # Remove ticks
        n.set_yticks([])
        if(nr < n_neurons-1):
            n.set_xticks([])
        # For some neurons, put their number on the label
        if(nr % 10 == 9):
            n.set_ylabel(nr+1)
        # Limit x axis to the duration and set the label
        n.set_xlim([0, duration])
    if sequence is not None:
        for nr, n in enumerate(ax):
            for t in sequence[nr]:
                n.axvline(t, color=color_seq)
    n.set_xlabel("time(s)")
    # Add common label "Neuron"
    fig.text(0.08, 0.5, 'Neuron', ha='center', va='center', rotation='vertical')
    plt.show()