from collections import Counter
import math
import tqdm
import numpy as np
from Bio.Seq import Seq

from scipy.stats import entropy, gaussian_kde

############################ Diversity metrics  ############################

def shannon_entropy(amino_acids, aa_list="ACDEFGHIKLMNPQRSTVWY"):
    """
    Calculate Shannon entropy in bits for the specified amino acid alphabet.
    :param amino_acids: sequence: observed amino acids or translated sequences.
    :param aa_list: str or sequence: alphabet included in the entropy sum.
    :return: float: entropy. The input must be nonempty; symbols outside aa_list contribute to the denominator but not the sum.
    """
    count = Counter(amino_acids)
    total = len(amino_acids)
    entropy = 0

    for aa in aa_list:
        p = count.get(aa, 0) / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def simpson_index(amino_acids):
    """
    Calculate Simpson diversity as one minus the sum of squared frequencies.
    :param amino_acids: sequence: observed amino acids or translated sequences.
    :return: float: diversity of the observed symbols.
    """
    count = Counter(amino_acids)
    total = len(amino_acids)
    return 1 - sum((freq / total) ** 2 for freq in count.values())


def reached_div(time_stop, time_step, times_array, seqs_array):
    """
    Summarize translated sequences reached before successive time points across trajectories.
    :param time_stop: float or False: end time; in simulation functions, enables a time stopping rule when truthy.
    :param time_step: float: interval between measurements.
    :param times_array: sequence of arrays: cumulative times for each trajectory, sorted in ascending order.
    :param seqs_array: sequence of sequences: nucleotide trajectories aligned with times_array.
    :return: dict: time to [mean number of distinct amino acid sequences, mean Simpson diversity]. Each trajectory must have a sequence before each sampled time.
    """
    time_start = 0
    measurements = dict()

    ## translate the portion reached strictly before each measurement time
    for t in tqdm.tqdm(np.arange(time_start + time_step, time_stop + time_step, time_step)):
        searched_aa_divSimpson = []
        len_searched_aa = []

        for el, seq_el in zip(times_array, seqs_array):
            smaller_idx = np.searchsorted(el, t, side='left')
            seqs_slice = seq_el[:smaller_idx]

            searched_aa = []

            for s_slice in seqs_slice:
                new_at = str(Seq(s_slice).translate())
                searched_aa.append(new_at)

            searched_aa_divSimpson.append(simpson_index(searched_aa))
            len_searched_aa.append(len(set(searched_aa)))

        measurements[t] = [np.mean(len_searched_aa), np.mean(searched_aa_divSimpson)]

    return measurements

def compute_divergence(x, y, kde = False, bins = 50, epsilon = False, base = 2):
    ## or epsilon=1e-12

    """
    Compare two distributions using shared histogram bins or kernel density estimates.
    :param x: array-like: first sample or x coordinates, according to the function.
    :param y: array-like: second sample or y coordinates, according to the function.
    :param kde: bool: use Gaussian kernel density estimates instead of histograms.
    :param bins: int or sequence: histogram bins passed to numpy.histogram.
    :param epsilon: float or False: pseudocount added to each histogram bin when truthy.
    :param base: float: logarithm base for divergence.
    :return: tuple[float, float, float]: KL(x || y), KL(y || x), and Jensen-Shannon divergence. Histogram edges are determined from x; values of y outside them are excluded.
    """
    if not kde:
        ## bin the data with shared bin edges
        counts_x, bin_edges = np.histogram(x, bins = bins, density = False)
        counts_y, _ = np.histogram(y, bins = bin_edges, density = False)

        ## convert to probabilities and add epsilon
        if epsilon:
            counts_x = counts_x.astype(float) + epsilon
            counts_y = counts_y.astype(float) + epsilon
        p = counts_x.astype(float) / sum(counts_x)
        q = counts_y.astype(float) / sum(counts_y)

    else:
        grid_size = 200
        kde_x = gaussian_kde(x)
        kde_y = gaussian_kde(y)

        ## create grid
        grid = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), grid_size)
        ## bin data
        p = kde_x(grid)
        q = kde_y(grid)

        ## normalize to get probabilities
        p = p / np.sum(p)
        q = q / np.sum(q)

    ## compute KL
    ### P || Q
    kl_pq = entropy(p, q, base=base)
    ### Q || P
    kl_qp = entropy(q, p, base=base)

    ## compute JS
    m = 0.5 * (p + q)
    js = 0.5 * entropy(p, m, base=base) + 0.5 * entropy(q, m, base=base)

    return kl_pq, kl_qp, js