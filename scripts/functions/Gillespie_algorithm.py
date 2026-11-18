import numpy as np
import math
import random
import tqdm

from collections import Counter

from . import mutation_matrix as mm

############################ Gillespie algorithm  ############################

def mut_cycle(current_seq, mutation_matrix, sequences):
    """
    Sample one mutation and its waiting time from a sparse rate matrix.
    :param current_seq: str: current nucleotide sequence.
    :param mutation_matrix: scipy sparse matrix: mutation rates indexed by sequence.
    :param sequences: sequence[str]: ordered sequences; indices must align with associated vectors or matrices.
    :return: tuple: selected one-nucleotide neighbor (str) and exponential waiting time (float). Rates must be positive and aligned with sequences.
    """
    current_idx = np.where(sequences == current_seq)[0]

    ## get sparse row
    row = mutation_matrix.getrow(current_idx)
    target_indices = row.indices
    target_rates = row.data

    ## sum of mutation rates excluding the diagonal
    rates_sum = sum(rate for idx, rate in zip(target_indices, target_rates) if idx != current_idx)

    ## randomly simulate the time of the next mutation
    tau = np.random.exponential(scale=1 / rates_sum)

    ## when mutation has happened, at which position of the sequence?

    ### sum of rates per possible sequence in distance = 1
    rates_per_neighbor = {}
    for idx, rate in zip(target_indices, target_rates):
        if idx == current_idx:
            continue
        neighbor_seq = sequences[idx]
        diffs = sum(1 for a, b in zip(current_seq, neighbor_seq) if a != b)
        if diffs == 1:
            rates_per_neighbor[neighbor_seq] = rate
    #print(rates_per_neighbor)

    ### normalize as probabilities
    rn_sum = sum(rates_per_neighbor.values())
    probabilities = [r / rn_sum for r in rates_per_neighbor.values()]
    probabilities = probabilities/sum(probabilities)

    ### given these rates, which particular mutation has occured
    chosen_seq = np.random.choice(list(rates_per_neighbor.keys()), p=probabilities)

    return chosen_seq, tau


def mut_trajectory(start_seq, n_repeats, mutation_matrix, mut_stop=False, time_stop=False):
    """
    Simulate repeated mutation trajectories starting from one k-mer.
    :param start_seq: str: initial k-mer.
    :param n_repeats: int: number of trajectories per starting sequence.
    :param mutation_matrix: scipy sparse matrix: mutation rates indexed by sequence.
    :param mut_stop: int or False: number of mutation events; takes precedence when truthy.
    :param time_stop: float or False: end time; in simulation functions, enables a time stopping rule when truthy.
    :return: tuple[np.ndarray, np.ndarray]: sequence trajectories and cumulative times, each including the starting state at time zero. mut_stop takes precedence over time_stop.
    """
    seqs_array = []
    times_array = []

    ## generate all possible k-mers
    kmers = mm.generate_all_sequences(len(start_seq))

    ## each replicate starts at time zero and follows one stopping rule
    for j in range(n_repeats):
        seqs = [start_seq]
        times = [0]
        current_seq = seqs[-1]

        if mut_stop:

            for i in range(mut_stop):
                next_seq, next_tau = mut_cycle(current_seq, mutation_matrix, kmers)
                seqs.append(next_seq)
                times.append(times[-1] + next_tau)
                current_seq = next_seq

        elif time_stop:

            while times[-1] < time_stop:
                next_seq, next_tau = mut_cycle(current_seq, mutation_matrix, kmers)
                if times[-1] + next_tau > time_stop:
                    break
                seqs.append(next_seq)
                times.append(times[-1] + next_tau)
                current_seq = next_seq

        seqs_array.append(seqs)
        times_array.append(times)

    seq_array = np.array(seqs_array, dtype=object)
    rate_array = np.array(times_array, dtype=object)

    return seq_array, rate_array


############################ Gillespie algorithm for k-mers ############################

def traj_counts(gene_kcdr, n_repeats, mutation_matrix, mut_stop=False, time_stop=False):
    """
    Count visits to sequences across simulated trajectories for all starting k-mers.
    :param gene_kcdr: iterable[str]: starting CDR k-mers.
    :param n_repeats: int: number of trajectories per starting sequence.
    :param mutation_matrix: scipy sparse matrix: mutation rates indexed by sequence.
    :param mut_stop: int or False: number of mutation events; takes precedence when truthy.
    :param time_stop: float or False: end time; in simulation functions, enables a time stopping rule when truthy.
    :return: dict[str, int]: total visits, including initial states.
    """
    trajectories = []
    for item in tqdm.tqdm(gene_kcdr):
        if mut_stop:
            seq_array, rate_array = mut_trajectory(item, n_repeats, mutation_matrix, mut_stop=mut_stop)
        elif time_stop:
            seq_array, rate_array = mut_trajectory(item, n_repeats, mutation_matrix, time_stop=time_stop)
        trajectories.extend(seq_array)

    melted_trajectories = np.concatenate(trajectories).ravel()

    unique_seqs, count_seqs = np.unique(melted_trajectories, return_counts=True)

    count_map = dict(zip(unique_seqs, count_seqs))
    return count_map

############################ Gillespie algorithm for long sequences ############################

def kmer_positions_preseeded(seq, sequences_idx, k, looped = False, overlapping = True):
    """
    Index starting positions of known k-mers in a sequence.
    :param seq: str: nucleotide sequence.
    :param sequences_idx: array-like[str]: ordered k-mers used to index position and rate arrays.
    :param k: int: k-mer length or number of requested eigenvalues, according to the function.
    :param looped: bool: include wrapping windows for overlapping k-mers.
    :param overlapping: bool: use every starting position; otherwise use nonoverlapping windows.
    :return: tuple: dict mapping every k-mer index to a list of positions, and an aligned integer count array. Circular handling applies only when overlapping is true.
    """
    m = len(sequences_idx)
    pos_lists = [[] for _ in range(m)]
    n = len(seq)

    ## collect window starts in the supplied k-mer order
    if m and k <= n:
        idx_of = {mer: i for i, mer in enumerate(sequences_idx)}
        if overlapping:
            if not looped:
                for i in range(n - k + 1):
                    mer = seq[i:i+k]
                    j = idx_of.get(mer)
                    if j is not None:
                        pos_lists[j].append(i)
            elif looped:
                for i in range(n):
                    if i + k < n + 1:
                        mer = seq[i:i + k]
                    else:
                        diff = n + 1 - k
                        first = seq[i:i + diff]
                        mer = first + seq[:k - len(first)]
                    j = idx_of.get(mer)
                    if j is not None:
                        pos_lists[j].append(i)
        else:
            for i in range(0, n - k + 1, k):
                mer = seq[i:i + k]
                j = idx_of.get(mer)
                if j is not None:
                    pos_lists[j].append(i)

    mers_weight = np.fromiter((len(lst) for lst in pos_lists), dtype=np.int64, count=m)
    mers_tomut = {i: lst for i, lst in enumerate(pos_lists)}
    return mers_tomut, mers_weight

def which_position(tomut_dict, tomut_weights, mutabilities):
    """
    Sample a k-mer occurrence and waiting time using occurrence-weighted mutabilities.
    :param tomut_dict: dict[int, list[int]]: k-mer index to starting positions.
    :param tomut_weights: np.ndarray: occurrence counts aligned with k-mer indices.
    :param mutabilities: np.ndarray: per-k-mer mutation rates aligned with counts.
    :return: tuple: waiting time, k-mer index, and starting position. The total rate must be positive.
    """
    rates = mutabilities*tomut_weights
    ## calculating time
    R = math.fsum(rates)
    tau = np.random.exponential(scale=1/R)

    ## normalize rates as probabilities
    probs = rates / R

    ## chosen mer
    chosen_mer = np.random.choice(list(tomut_dict.keys()), p=probs)
    positions = tomut_dict[chosen_mer]
    chosen_position = np.random.choice(positions)

    return tau, chosen_mer, chosen_position

def which_mutation(chosen_mer, muts_array):
    """
    Sample a replacement nucleotide from the last four columns of a mutation table row.
    :param chosen_mer: int: selected k-mer index.
    :param muts_array: np.ndarray: mutation table with A, C, G, T weights in its last four columns.
    :return: str: chosen nucleotide, ordered A, C, G, T in the input weights.
    """
    probs = list(muts_array[chosen_mer][-4:]/math.fsum(muts_array[chosen_mer][-4:]))
    chosen_mut = np.random.choice(['A', 'C', 'G', 'T'], p = probs)
    return chosen_mut

def remove_value(d, key, value):
    """
    Remove the first occurrence of a value from a dictionary list if present.
    :param d: dict: mapping from keys to lists.
    :param key: object: dictionary key.
    :param value: object: value to remove.
    :return: None: the dictionary is modified in place.
    """
    lst = d.get(key)
    if lst is None:
        return
    try:
        lst.remove(value)
    except ValueError:
        pass

def resulting_dict(sequences_idx, chosen_nt, chosen_mer, chosen_position, tomut_dict, tomut_weights, len_tomut_seq):
    """
    Update affected five-mer positions after a central nucleotide mutation.
    :param sequences_idx: array-like[str]: ordered k-mers used to index position and rate arrays.
    :param chosen_nt: str: replacement nucleotide.
    :param chosen_mer: int: selected k-mer index.
    :param chosen_position: int: start position of the chosen five-mer.
    :param tomut_dict: dict[int, list[int]]: k-mer index to starting positions.
    :param tomut_weights: np.ndarray: occurrence counts aligned with k-mer indices.
    :param len_tomut_seq: int: length of the sequence being mutated.
    :return: tuple: updated position dictionary and count array, both modified in place. This routine assumes five-mers and linear sequence boundaries.
    """

    ## update all sequences overlapping with chosen nucleotide
    update_pos = range(chosen_position-2, chosen_position+3)
    for position in update_pos:
        if 0 <= position and position + 5 <= len_tomut_seq:
            ### k-mer at this position
            update_mer_idx = next((k for k, v in tomut_dict.items() if position in v), None)
            update_mer = sequences_idx[update_mer_idx]

            ### update k-mer: change chosen nucleotide
            update_nt = chosen_position - position + 2
            update_mer_list = list(update_mer)
            update_mer_list[update_nt] = chosen_nt

            ### remove previous version from weights
            tomut_weights[update_mer_idx] -= 1
            remove_value(tomut_dict, update_mer_idx, position)

            ### add updated mer to position mapping
            update_mer_idx = np.where(sequences_idx == ''.join(update_mer_list))[0][0]
            tomut_dict[update_mer_idx].append(position)
            tomut_weights[update_mer_idx] += 1

    return tomut_dict, tomut_weights

############################ Random AA sequence generation ############################

def random_nt_sequence(length: "int") -> "str":
    """
    Generate a random nucleotide sequence.
    :param length: int: desired sequence length.
    :return: str: sequence of the requested length, sampled from A, C, G and T.
    """
    nts = 'ACGT'
    return ''.join(random.choice(nts) for _ in range(length))