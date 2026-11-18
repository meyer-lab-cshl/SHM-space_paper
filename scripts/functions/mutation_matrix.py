import numpy as np
import tqdm
import itertools
from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.optimize import linear_sum_assignment

from Bio.Data import CodonTable

############################ mutation matrix ############################

def generate_all_sequences(l: "int") -> "np.ndarray":
    """
    Enumerate all DNA sequences of a given length in ATGC product order.
    :param l: int: sequence length.
    :return: np.ndarray: 4**l sequences.
    """

    return np.array(["".join(seq) for seq in itertools.product("ATGC", repeat=l)])


def cycled_context(seq: "str", index: "int", l_context: "int") -> "str":
    """
    Extract a centered context using circular sequence boundaries.
    :param seq: str: nucleotide sequence.
    :param index: int: center position.
    :param l_context: int: number of flanking nucleotides on each side.
    :return: str: context of length 2 * l_context + 1.
    """

    context = "".join(seq[(index + i - l_context) % len(seq)] for i in range(l_context * 2 + 1))
    return context


def compute_mutmatrix(sequences: "list[str]", muts: "DataFrame", l_seq: "int", l_context: "int") -> "sp.csr_matrix":
    """
    Build a sparse continuous-time mutation rate matrix using circular contexts.
    :param sequences: sequence[str]: ordered sequences; indices must align with associated vectors or matrices.
    :param muts: pandas.DataFrame: Motif, Mutability, To A, To C, To G and To T columns.
    :param l_seq: int: number of nucleotide positions per sequence.
    :param l_context: int: number of flanking nucleotides on each side.
    :return: scipy.sparse.csr_matrix: rates between sequences, with negative row sums on the diagonal. The sequence collection must contain every single-nucleotide target and the table every context.
    """

    ## assign an index to each sequence
    seq_index = {seq: i for i, seq in enumerate(sequences)}

    ## a sparse matrix in LIL format for efficient row-based modifications
    size = len(sequences)
    mutation_matrix = sp.lil_matrix((size, size), dtype=np.float64)

    for seq in tqdm.tqdm(sequences, desc="Generating transition matrix"):
        row_index = seq_index[seq]
        ## sum of the row
        row_sum = 0

        for i in range(l_seq):
            context = cycled_context(seq, i, l_context)

            ## corresponding motif in the mutability table
            row = muts[muts['Motif'] == context].iloc[0]
            mutability = row['Mutability']
            mutation_probs = {'A': row['To A'], 'C': row['To C'], 'G': row['To G'], 'T': row['To T']}

            ## original nucleotide in the sequence
            original_nt = seq[i]

            ## iterate through possible mutations
            for new_nt, prob in mutation_probs.items():
                if new_nt != original_nt:
                    ## mutated sequence
                    mutated_seq = seq[:i] + new_nt + seq[i + 1:]
                    ## index of mutated sequence
                    mutated_index = seq_index[mutated_seq]

                    ## mutation probability
                    mutation_prob = mutability * prob
                    mutation_matrix[row_index, mutated_index] += mutation_prob

                    ## update row sum
                    row_sum += mutation_prob

        ## diagonal element equals to negative row sum
        mutation_matrix[row_index, row_index] = -row_sum

    ## CSR format
    return mutation_matrix.tocsr()

def k_eigenvectors_old(k: "int", mutation_matrix_sp: "sp.csc_matrix", sequences: "list[str]", which: "str" = 'LR') -> "tuple[list, list, list, list]":
    """
    Calculate the legacy stationary vector and two scaled diffusion axes.
    :param k: int: k-mer length or number of requested eigenvalues, according to the function.
    :param mutation_matrix_sp: scipy sparse matrix: square mutation rate matrix.
    :param sequences: sequence[str]: ordered sequences; indices must align with associated vectors or matrices.
    :param which: str: eigenvalue selection passed to scipy.sparse.linalg.eigs.
    :return: tuple: right eigenvalues, normalized first transpose eigenvector, and two scaled right eigenvectors. This legacy method assumes eigensolver ordering and uses the all-A sequence to orient axes.
    """

    right_eigvals, right_eigenvectors = eigs(mutation_matrix_sp, k=k, which=which)
    _, left_eigenvectors = eigs(mutation_matrix_sp.transpose(), k=k, which=which)

    stat_freq = left_eigenvectors[:, 0]

    v1 = right_eigenvectors[:, 1]
    v2 = right_eigenvectors[:, 2]

    ## transform diffusion axes to timescale units
    v1 = v1 * (1/np.sqrt(-right_eigvals[1]))
    v2 = v2 * (1 / np.sqrt(-right_eigvals[2]))

    ref_seq = 'A'*len(sequences[0])
    aaa_idx = np.where(sequences == ref_seq)[0][0]
    if v1[aaa_idx] < 0:
        v1 = v1 / -1
    if v2[aaa_idx] < 0:
        v2 = v2 / -1

    stat_freq_norm = stat_freq / np.sum(stat_freq)

    return right_eigvals, stat_freq_norm, v1, v2

def k_eigenvectors_phased(k: 'int', mutation_matrix_sp: 'sp.spmatrix', sequences: 'list[str]',
                          which: 'str' = 'LR', eig_tol: 'float' = 1e-8) -> 'tuple[list, list, list, list]':
    """
    Calculate stationary frequencies and phased, timescale-scaled diffusion axes.
    :param k: int: k-mer length or number of requested eigenvalues, according to the function.
    :param mutation_matrix_sp: scipy sparse matrix: square mutation rate matrix.
    :param sequences: sequence[str]: ordered sequences; indices must align with associated vectors or matrices.
    :param which: str: eigenvalue selection passed to scipy.sparse.linalg.eigs.
    :param eig_tol: float: absolute and relative tolerance for grouping eigenvalues.
    :return: tuple: ordered right eigenvalues, normalized stationary vector, and the first two nonstationary right eigenvectors scaled by square roots of decay times. Requires at least three eigenvectors and an all-A sequence.
    """

    mutation_matrix_sp = mutation_matrix_sp.tocsc()

    ## right and transpose eigenvectors
    right_eigvals, right_eigenvectors = eigs(mutation_matrix_sp, k=k, which=which)
    trans_eigvals, trans_eigenvectors = eigs(mutation_matrix_sp.transpose(), k=k, which=which)

    ## match left and right eigenvectors by eigenvalue
    rows, cols = linear_sum_assignment(np.abs(right_eigvals[:, None] - trans_eigvals[None, :]))
    matched_cols = dict(zip(rows, cols))
    left_eigenvectors = np.column_stack([np.conj(trans_eigenvectors[:, matched_cols[j]]) for j in range(len(right_eigvals))])

    ## order eigenvalues from largest to smallest real part
    order = np.argsort(right_eigvals.real)[::-1]
    right_eigvals = right_eigvals[order]
    right_eigenvectors = right_eigenvectors[:, order]
    left_eigenvectors = left_eigenvectors[:, order]

    ## stationary distribution
    stat_freq = left_eigenvectors[:, 0]
    ## normalization to sum up to 1
    stat_freq_norm = stat_freq / np.sum(stat_freq)

    ## normalize right eigenvectors so r* D_pi r = 1
    for j in range(right_eigenvectors.shape[1]):
        norm = np.sqrt(np.vdot(right_eigenvectors[:, j], stat_freq_norm * right_eigenvectors[:, j] ).real)
        right_eigenvectors[:, j] /= norm

    ## normalize left eigenvectors so l* r = 1 and construct biorthogonal bases for repeated eigenvalues
    processed = set()
    ### for every eigenvalue
    for j, eigval in enumerate(right_eigvals):
        ### since some eigenvalues are repeated, it could have been already processed
        if j in processed:
            continue
        ### find all eigenvalues numerically similar to a chosen eigenvalue
        group = np.where(np.isclose(right_eigvals, eigval, rtol=eig_tol, atol=eig_tol))[0]
        processed.update(group)
        ### extract right add left eigenvectors associated with that eigenvalue
        right_group = right_eigenvectors[:, group]
        left_group = left_eigenvectors[:, group]
        ### compute pairwise inner products between left and right eigenvectors
        overlap = left_group.conj().T @ right_group
        ### selected left eigenvectors become biorthogonal to the right eigenvectors
        left_eigenvectors[:, group] = left_group @ np.linalg.inv(overlap).conj().T

    ## set eigenvector phase using the all-A sequence
    ref_seq = 'A' * len(sequences[0])
    aaa_idx = np.where(np.asarray(sequences) == ref_seq)[0][0]
    for j in range(len(right_eigvals)):
        if j != 0:
            phase = np.angle(right_eigenvectors[aaa_idx, j])
            rotation = np.exp(-1j * phase)

            right_eigenvectors[:, j] *= rotation
            left_eigenvectors[:, j] *= rotation

    ## transform eigenvectors to timescale units
    tau1 = 1 / (-right_eigvals[1].real)
    tau2 = 1 / (-right_eigvals[2].real)
    v1 = right_eigenvectors[:, 1] * np.sqrt(tau1)
    v2 = right_eigenvectors[:, 2] * np.sqrt(tau2)

    return right_eigvals, stat_freq_norm, v1, v2


############################ For codons matrix ############################

def kmer_to_codon(codons: "list", kmer: "int", p1: "int", p2: "int") -> "dict":
    """
    Map each generated k-mer index to the index of its codon slice.
    :param codons: sequence[str]: ordered codon labels.
    :param kmer: int: k-mer length.
    :param p1: int: inclusive start of the slice.
    :param p2: int: exclusive end of the slice.
    :return: dict[int, int]: k-mer index to codon index; the slice must occur in codons.
    """

    kmer_to_codon_idx = {}

    ## generate all possible sequences of length 3 and create an index for them
    codon_to_idx = {codon: i for i, codon in enumerate(codons)}

    ## extracts the codon from positions p1 to p2 of the 8-mer and assigns its index
    for i, mer in enumerate(generate_all_sequences(kmer)):
        codon = mer[p1:p2]
        kmer_to_codon_idx[i] = codon_to_idx[codon]
    return kmer_to_codon_idx


def sum_codons(codons: "list[str]", sequences: "list[str]", vector: "np.array[float]",
               vector_type: "str", p1: "int", p2: "int")-> "list":
    """
    Aggregate a k-mer vector by the codon at the selected slice.
    :param codons: sequence[str]: ordered codon labels.
    :param sequences: sequence[str]: ordered sequences; indices must align with associated vectors or matrices.
    :param vector: array-like: values aligned with the input sequence or codon order.
    :param vector_type: str: 'average' divides by group size; other values return sums.
    :param p1: int: inclusive start of the slice.
    :param p2: int: exclusive end of the slice.
    :return: np.ndarray: summed values, or averages if vector_type is average. Values must follow generate_all_sequences order.
    """
    ## vector_type: 'sum' or 'average'

    idx = kmer_to_codon(codons = codons, kmer = len(sequences[0]), p1 = p1, p2 = p2)

    counts = list(idx.values()).count(0)

    codon_vector = np.zeros(len(codons))

    for i, value in enumerate(vector):
        codon_index = idx[i]
        codon_vector[codon_index] += value

    if vector_type == 'average':
        codon_vector = codon_vector / counts

    return codon_vector

############################ For any n-mer matrix ############################

def kmer_to_nmer(nmers: "list", kmer: "int", p1: "int", p2: "int") -> "dict":
    """
    Map each generated k-mer index to the index of its n-mer slice.
    :param nmers: sequence[str]: ordered n-mer labels.
    :param kmer: int: k-mer length.
    :param p1: int: inclusive start of the slice.
    :param p2: int: exclusive end of the slice.
    :return: dict[int, int]: k-mer index to n-mer index; the slice must occur in nmers.
    """

    kmer_to_nmer_idx = {}

    ## generate all possible sequences of length 3 and create an index for them
    nmer_to_idx = {nmer: i for i, nmer in enumerate(nmers)}

    ## extracts the codon from positions p1 to p2 of the 8-mer and assigns its index
    for i, mer in enumerate(generate_all_sequences(kmer)):
        nmer = mer[p1:p2]
        kmer_to_nmer_idx[i] = nmer_to_idx[nmer]
    return kmer_to_nmer_idx


def sum_nmers(nmers: "list[str]", sequences: "list[str]", vector: "np.array[float]",
               vector_type: "str", p1: "int", p2: "int")-> "list":
    """
    Aggregate a k-mer vector by the n-mer at the selected slice.
    :param nmers: sequence[str]: ordered n-mer labels.
    :param sequences: sequence[str]: ordered sequences; indices must align with associated vectors or matrices.
    :param vector: array-like: values aligned with the input sequence or codon order.
    :param vector_type: str: 'average' divides by group size; other values return sums.
    :param p1: int: inclusive start of the slice.
    :param p2: int: exclusive end of the slice.
    :return: np.ndarray: summed values, or averages if vector_type is average. Values must follow generate_all_sequences order.
    """
    ## vector_type: 'sum' or 'average'

    idx = kmer_to_nmer(nmers = nmers, kmer = len(sequences[0]), p1 = p1, p2 = p2)

    counts = list(idx.values()).count(0)

    nmer_vector = np.zeros(len(nmers))

    for i, value in enumerate(vector):
        nmer_index = idx[i]
        nmer_vector[nmer_index] += value

    if vector_type == 'average':
        nmer_vector = nmer_vector / counts

    return nmer_vector


############################ For amino acid matrix ############################

def codon_to_aa(codons: "list[str]", include_stop: "bool" =True, separate_ser = False) -> "tuple[dict, list[str]]":
    """
    Map codon indices to sorted amino acid indices using the standard genetic code.
    :param codons: sequence[str]: ordered codon labels.
    :param include_stop: bool: include stop codons as *.
    :param separate_ser: bool: separate serine codons into S2 and S4 groups.
    :return: tuple[dict, list[str]]: codon-index mapping and amino acid labels. With separate_ser false, serine codons are currently omitted from the mapping; stops are omitted if include_stop is false.
    """
    # Maps codons to amino acid indices

    std_table = CodonTable.unambiguous_dna_by_name["Standard"]
    forward = std_table.forward_table
    stop_codons = std_table.stop_codons
    s2_codons = ['AGT', 'AGC']
    s4_codons = ['TCT', 'TCC', 'TCA', 'TCG']

    # Build amino acid list
    aa_set = set(forward.values())
    if include_stop:
        aa_set.add('*')
    if separate_ser:
        aa_set.add('S2')
        aa_set.add('S4')
        aa_set.remove('S')

    aa_list = sorted(aa_set)
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}

    codon_to_aa_idx = {}
    for i, codon in enumerate(codons):
        if codon in stop_codons:
            if include_stop:
                codon_to_aa_idx[i] = aa_to_idx['*']
        elif codon in s2_codons:
            if separate_ser:
                codon_to_aa_idx[i] = aa_to_idx['S2']
        elif codon in s4_codons:
            if separate_ser:
                codon_to_aa_idx[i] = aa_to_idx['S4']
        else:
            aa = forward.get(codon, None)
            if aa:
                codon_to_aa_idx[i] = aa_to_idx[aa]

    return codon_to_aa_idx, aa_list


def sum_aas(codons: "list[str]", vector: "np.array[float]",
            vector_type: "str" ='sum', include_stop: "bool" =True, separate_ser = False) -> "tuple[list[float], list[str]]":
    """
    Aggregate codon values by amino acid using codon_to_aa.
    :param codons: sequence[str]: ordered codon labels.
    :param vector: array-like: values aligned with the input sequence or codon order.
    :param vector_type: str: 'average' divides by group size; other values return sums.
    :param include_stop: bool: include stop codons as *.
    :param separate_ser: bool: separate serine codons into S2 and S4 groups.
    :return: tuple[np.ndarray, list[str]]: sums or averages and corresponding amino acid labels. Unmapped codons are omitted.
    """

    codon_to_aa_idx, aa_list = codon_to_aa(codons, include_stop=include_stop, separate_ser=separate_ser)
    aa_vector = np.zeros(len(aa_list))

    count_vector = np.zeros(len(aa_list))

    for i, value in enumerate(vector):
        if i in codon_to_aa_idx:
            aa_index = codon_to_aa_idx[i]
            aa_vector[aa_index] += value
            if vector_type == 'average':
                count_vector[aa_index] += 1

    if vector_type == 'average':
        with np.errstate(divide='ignore', invalid='ignore'):
            aa_vector = np.divide(
                aa_vector, count_vector,
                out=np.zeros_like(aa_vector),
                where=count_vector != 0
            )

    return aa_vector, aa_list

############################ Graph representation ############################

def connectivity(sequences):

    """
    Find sequence pairs that share a masked position.
    :param sequences: sequence[str]: ordered sequences; indices must align with associated vectors or matrices.
    :return: dict[int, set[int]]: sequence indices and their neighboring indices. Input sequences should have equal lengths.
    """

    ## sort sequences into "buckets" - sequences that are the same after a particular position is masked
    buckets = defaultdict(list)
    for s in range(len(sequences)):
        seq = sequences[s]
        for i in range(len(sequences[0])):
            key = seq[:i] + "*" + seq[i+1:]
            buckets[key].append(s)

    ## build adjacency from shared bucket
    adjacent = {s: set() for s in range(len(sequences))}
    ### iterating through lists of similar sequences (members of one bucket)
    for members in tqdm.tqdm(buckets.values()):
        for i in range(len(members)):
            si = members[i]
            for j in range(i + 1, len(members)):
                sj = members[j]
                if si != sj:
                    adjacent[si].add(sj)
                    adjacent[sj].add(si)

    return adjacent