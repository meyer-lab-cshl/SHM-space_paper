# Function reference

These functions support the analysis of somatic hypermutation in B-cell receptor sequence space. They are used to generate nucleotide k-mers, build context-dependent mutation matrices, and calculate stationary frequencies and diffusion axes to describe the mutational landscape. Additionally, they are used to aggregate k-mer results to codons and amino acids, extract CDR1 and CDR2 k-mers from V genes, and simulate mutation trajectories with the Gillespie algorithm. Plotting helpers display the resulting sequence-space networks, trajectories, and frequency comparisons.

The reference below describes these helpers.

## CDR k-mer generation

### `cdr_kmers_generation.kmers(sequence, kmer)`

Collect distinct, complete k-mers without N from a sequence.

**Parameters:**

- **sequence** (*str*) – nucleotide sequence.
- **kmer** (*int*) – k-mer length.

**Returns:** distinct k-mers in unspecified order.

**Return type:** `list[str]`

### `cdr_kmers_generation.kmer_gene_generation(cdrs_table, k, category, gene_id, seq_to_idx, printed=False, downsample=False, unique=False)`

Collect CDR1 and CDR2 k-mers for a gene or gene class.

**Parameters:**

- **cdrs_table** (*pandas.DataFrame*) – rows with gene_class, gene, nt_sequence and CDR columns.
- **k** (*int*) – k-mer length or number of requested eigenvalues, according to the function.
- **category** (*str*) – 'gene_class' selects the gene_class column; any other value selects gene.
- **gene_id** (*str*) – gene or gene-class identifier to select.
- **seq_to_idx** (*dict[str, int]*) – mapping from k-mers to their indices.
- **printed** (*bool*) – whether to print unique k-mer counts.
- **downsample** (*int or False*) – number of distinct genes to sample without replacement.
- **unique** (*bool*) – whether to deduplicate each CDR k-mer list.

**Returns:** number of genes, CDR1 k-mers, CDR2 k-mers, and their respective index lists. Repeated k-mers are retained unless unique is true.

**Return type:** `tuple`

## Mutation matrices and sequence space

### `mutation_matrix.generate_all_sequences(l)`

Enumerate all DNA sequences of a given length in ATGC product order.

**Parameters:**

- **l** (*int*) – sequence length.

**Returns:** 4**l sequences.

**Return type:** `np.ndarray`

### `mutation_matrix.cycled_context(seq, index, l_context)`

Extract a centered context using circular sequence boundaries.

**Parameters:**

- **seq** (*str*) – nucleotide sequence.
- **index** (*int*) – center position.
- **l_context** (*int*) – number of flanking nucleotides on each side.

**Returns:** context of length 2 * l_context + 1.

**Return type:** `str`

### `mutation_matrix.compute_mutmatrix(sequences, muts, l_seq, l_context)`

Build a sparse continuous-time mutation rate matrix using circular contexts.

**Parameters:**

- **sequences** (*sequence[str]*) – ordered sequences; indices must align with associated vectors or matrices.
- **muts** (*pandas.DataFrame*) – Motif, Mutability, To A, To C, To G and To T columns.
- **l_seq** (*int*) – number of nucleotide positions per sequence.
- **l_context** (*int*) – number of flanking nucleotides on each side.

**Returns:** rates between sequences, with negative row sums on the diagonal. The sequence collection must contain every single-nucleotide target and the table every context.

**Return type:** `scipy.sparse.csr_matrix`

### `mutation_matrix.k_eigenvectors_old(k, mutation_matrix_sp, sequences, which='LR')`

Calculate the legacy stationary vector and two scaled diffusion axes.

**Parameters:**

- **k** (*int*) – k-mer length or number of requested eigenvalues, according to the function.
- **mutation_matrix_sp** (*scipy sparse matrix*) – square mutation rate matrix.
- **sequences** (*sequence[str]*) – ordered sequences; indices must align with associated vectors or matrices.
- **which** (*str*) – eigenvalue selection passed to scipy.sparse.linalg.eigs.

**Returns:** right eigenvalues, normalized first transpose eigenvector, and two scaled right eigenvectors. This legacy method assumes eigensolver ordering and uses the all-A sequence to orient axes.

**Return type:** `tuple`

### `mutation_matrix.k_eigenvectors_phased(k, mutation_matrix_sp, sequences, which='LR', eig_tol=1e-08)`

Calculate stationary frequencies and phased, timescale-scaled diffusion axes.

**Parameters:**

- **k** (*int*) – k-mer length or number of requested eigenvalues, according to the function.
- **mutation_matrix_sp** (*scipy sparse matrix*) – square mutation rate matrix.
- **sequences** (*sequence[str]*) – ordered sequences; indices must align with associated vectors or matrices.
- **which** (*str*) – eigenvalue selection passed to scipy.sparse.linalg.eigs.
- **eig_tol** (*float*) – absolute and relative tolerance for grouping eigenvalues.

**Returns:** ordered right eigenvalues, normalized stationary vector, and the first two nonstationary right eigenvectors scaled by square roots of decay times. Requires at least three eigenvectors and an all-A sequence.

**Return type:** `tuple`

### `mutation_matrix.kmer_to_codon(codons, kmer, p1, p2)`

Map each generated k-mer index to the index of its codon slice.

**Parameters:**

- **codons** (*sequence[str]*) – ordered codon labels.
- **kmer** (*int*) – k-mer length.
- **p1** (*int*) – inclusive start of the slice.
- **p2** (*int*) – exclusive end of the slice.

**Returns:** k-mer index to codon index; the slice must occur in codons.

**Return type:** `dict[int, int]`

### `mutation_matrix.sum_codons(codons, sequences, vector, vector_type, p1, p2)`

Aggregate a k-mer vector by the codon at the selected slice.

**Parameters:**

- **codons** (*sequence[str]*) – ordered codon labels.
- **sequences** (*sequence[str]*) – ordered sequences; indices must align with associated vectors or matrices.
- **vector** (*array-like*) – values aligned with the input sequence or codon order.
- **vector_type** (*str*) – 'average' divides by group size; other values return sums.
- **p1** (*int*) – inclusive start of the slice.
- **p2** (*int*) – exclusive end of the slice.

**Returns:** summed values, or averages if vector_type is average. Values must follow generate_all_sequences order.

**Return type:** `np.ndarray`

### `mutation_matrix.kmer_to_nmer(nmers, kmer, p1, p2)`

Map each generated k-mer index to the index of its n-mer slice.

**Parameters:**

- **nmers** (*sequence[str]*) – ordered n-mer labels.
- **kmer** (*int*) – k-mer length.
- **p1** (*int*) – inclusive start of the slice.
- **p2** (*int*) – exclusive end of the slice.

**Returns:** k-mer index to n-mer index; the slice must occur in nmers.

**Return type:** `dict[int, int]`

### `mutation_matrix.sum_nmers(nmers, sequences, vector, vector_type, p1, p2)`

Aggregate a k-mer vector by the n-mer at the selected slice.

**Parameters:**

- **nmers** (*sequence[str]*) – ordered n-mer labels.
- **sequences** (*sequence[str]*) – ordered sequences; indices must align with associated vectors or matrices.
- **vector** (*array-like*) – values aligned with the input sequence or codon order.
- **vector_type** (*str*) – 'average' divides by group size; other values return sums.
- **p1** (*int*) – inclusive start of the slice.
- **p2** (*int*) – exclusive end of the slice.

**Returns:** summed values, or averages if vector_type is average. Values must follow generate_all_sequences order.

**Return type:** `np.ndarray`

### `mutation_matrix.codon_to_aa(codons, include_stop=True, separate_ser=False)`

Map codon indices to sorted amino acid indices using the standard genetic code.

**Parameters:**

- **codons** (*sequence[str]*) – ordered codon labels.
- **include_stop** (*bool*) – include stop codons as *.
- **separate_ser** (*bool*) – separate serine codons into S2 and S4 groups.

**Returns:** codon-index mapping and amino acid labels. With separate_ser false, serine codons are currently omitted from the mapping; stops are omitted if include_stop is false.

**Return type:** `tuple[dict, list[str]]`

### `mutation_matrix.sum_aas(codons, vector, vector_type='sum', include_stop=True, separate_ser=False)`

Aggregate codon values by amino acid using codon_to_aa.

**Parameters:**

- **codons** (*sequence[str]*) – ordered codon labels.
- **vector** (*array-like*) – values aligned with the input sequence or codon order.
- **vector_type** (*str*) – 'average' divides by group size; other values return sums.
- **include_stop** (*bool*) – include stop codons as *.
- **separate_ser** (*bool*) – separate serine codons into S2 and S4 groups.

**Returns:** sums or averages and corresponding amino acid labels. Unmapped codons are omitted.

**Return type:** `tuple[np.ndarray, list[str]]`

### `mutation_matrix.connectivity(sequences)`

Find sequence pairs that share a masked position.

**Parameters:**

- **sequences** (*sequence[str]*) – ordered sequences; indices must align with associated vectors or matrices.

**Returns:** sequence indices and their neighboring indices. Input sequences should have equal lengths.

**Return type:** `dict[int, set[int]]`

## Gillespie simulation

### `Gillespie_algorithm.mut_cycle(current_seq, mutation_matrix, sequences)`

Sample one mutation and its waiting time from a sparse rate matrix.

**Parameters:**

- **current_seq** (*str*) – current nucleotide sequence.
- **mutation_matrix** (*scipy sparse matrix*) – mutation rates indexed by sequence.
- **sequences** (*sequence[str]*) – ordered sequences; indices must align with associated vectors or matrices.

**Returns:** selected one-nucleotide neighbor (str) and exponential waiting time (float). Rates must be positive and aligned with sequences.

**Return type:** `tuple`

### `Gillespie_algorithm.mut_trajectory(start_seq, n_repeats, mutation_matrix, mut_stop=False, time_stop=False)`

Simulate repeated mutation trajectories starting from one k-mer.

**Parameters:**

- **start_seq** (*str*) – initial k-mer.
- **n_repeats** (*int*) – number of trajectories per starting sequence.
- **mutation_matrix** (*scipy sparse matrix*) – mutation rates indexed by sequence.
- **mut_stop** (*int or False*) – number of mutation events; takes precedence when truthy.
- **time_stop** (*float or False*) – end time; in simulation functions, enables a time stopping rule when truthy.

**Returns:** sequence trajectories and cumulative times, each including the starting state at time zero. mut_stop takes precedence over time_stop.

**Return type:** `tuple[np.ndarray, np.ndarray]`

### `Gillespie_algorithm.traj_counts(gene_kcdr, n_repeats, mutation_matrix, mut_stop=False, time_stop=False)`

Count visits to sequences across simulated trajectories for all starting k-mers.

**Parameters:**

- **gene_kcdr** (*iterable[str]*) – starting CDR k-mers.
- **n_repeats** (*int*) – number of trajectories per starting sequence.
- **mutation_matrix** (*scipy sparse matrix*) – mutation rates indexed by sequence.
- **mut_stop** (*int or False*) – number of mutation events; takes precedence when truthy.
- **time_stop** (*float or False*) – end time; in simulation functions, enables a time stopping rule when truthy.

**Returns:** total visits, including initial states.

**Return type:** `dict[str, int]`

### `Gillespie_algorithm.kmer_positions_preseeded(seq, sequences_idx, k, looped=False, overlapping=True)`

Index starting positions of known k-mers in a sequence.

**Parameters:**

- **seq** (*str*) – nucleotide sequence.
- **sequences_idx** (*array-like[str]*) – ordered k-mers used to index position and rate arrays.
- **k** (*int*) – k-mer length or number of requested eigenvalues, according to the function.
- **looped** (*bool*) – include wrapping windows for overlapping k-mers.
- **overlapping** (*bool*) – use every starting position; otherwise use nonoverlapping windows.

**Returns:** dict mapping every k-mer index to a list of positions, and an aligned integer count array. Circular handling applies only when overlapping is true.

**Return type:** `tuple`

### `Gillespie_algorithm.which_position(tomut_dict, tomut_weights, mutabilities)`

Sample a k-mer occurrence and waiting time using occurrence-weighted mutabilities.

**Parameters:**

- **tomut_dict** (*dict[int, list[int]]*) – k-mer index to starting positions.
- **tomut_weights** (*np.ndarray*) – occurrence counts aligned with k-mer indices.
- **mutabilities** (*np.ndarray*) – per-k-mer mutation rates aligned with counts.

**Returns:** waiting time, k-mer index, and starting position. The total rate must be positive.

**Return type:** `tuple`

### `Gillespie_algorithm.which_mutation(chosen_mer, muts_array)`

Sample a replacement nucleotide from the last four columns of a mutation table row.

**Parameters:**

- **chosen_mer** (*int*) – selected k-mer index.
- **muts_array** (*np.ndarray*) – mutation table with A, C, G, T weights in its last four columns.

**Returns:** chosen nucleotide, ordered A, C, G, T in the input weights.

**Return type:** `str`

### `Gillespie_algorithm.remove_value(d, key, value)`

Remove the first occurrence of a value from a dictionary list if present.

**Parameters:**

- **d** (*dict*) – mapping from keys to lists.
- **key** (*object*) – dictionary key.
- **value** (*object*) – value to remove.

**Returns:** the dictionary is modified in place.

**Return type:** `None`

### `Gillespie_algorithm.resulting_dict(sequences_idx, chosen_nt, chosen_mer, chosen_position, tomut_dict, tomut_weights, len_tomut_seq)`

Update affected five-mer positions after a central nucleotide mutation.

**Parameters:**

- **sequences_idx** (*array-like[str]*) – ordered k-mers used to index position and rate arrays.
- **chosen_nt** (*str*) – replacement nucleotide.
- **chosen_mer** (*int*) – selected k-mer index.
- **chosen_position** (*int*) – start position of the chosen five-mer.
- **tomut_dict** (*dict[int, list[int]]*) – k-mer index to starting positions.
- **tomut_weights** (*np.ndarray*) – occurrence counts aligned with k-mer indices.
- **len_tomut_seq** (*int*) – length of the sequence being mutated.

**Returns:** updated position dictionary and count array, both modified in place. This routine assumes five-mers and linear sequence boundaries.

**Return type:** `tuple`

### `Gillespie_algorithm.random_nt_sequence(length)`

Generate a random nucleotide sequence.

**Parameters:**

- **length** (*int*) – desired sequence length.

**Returns:** sequence of the requested length, sampled from A, C, G and T.

**Return type:** `str`

## Diversity and divergence

### `diversity.shannon_entropy(amino_acids, aa_list='ACDEFGHIKLMNPQRSTVWY')`

Calculate Shannon entropy in bits for the specified amino acid alphabet.

**Parameters:**

- **amino_acids** (*sequence*) – observed amino acids or translated sequences.
- **aa_list** (*str or sequence*) – alphabet included in the entropy sum.

**Returns:** entropy. The input must be nonempty; symbols outside aa_list contribute to the denominator but not the sum.

**Return type:** `float`

### `diversity.simpson_index(amino_acids)`

Calculate Simpson diversity as one minus the sum of squared frequencies.

**Parameters:**

- **amino_acids** (*sequence*) – observed amino acids or translated sequences.

**Returns:** diversity of the observed symbols.

**Return type:** `float`

### `diversity.reached_div(time_stop, time_step, times_array, seqs_array)`

Summarize translated sequences reached before successive time points across trajectories.

**Parameters:**

- **time_stop** (*float or False*) – end time; in simulation functions, enables a time stopping rule when truthy.
- **time_step** (*float*) – interval between measurements.
- **times_array** (*sequence of arrays*) – cumulative times for each trajectory, sorted in ascending order.
- **seqs_array** (*sequence of sequences*) – nucleotide trajectories aligned with times_array.

**Returns:** time to [mean number of distinct amino acid sequences, mean Simpson diversity]. Each trajectory must have a sequence before each sampled time.

**Return type:** `dict`

### `diversity.compute_divergence(x, y, kde=False, bins=50, epsilon=False, base=2)`

Compare two distributions using shared histogram bins or kernel density estimates.

**Parameters:**

- **x** (*array-like*) – first sample or x coordinates, according to the function.
- **y** (*array-like*) – second sample or y coordinates, according to the function.
- **kde** (*bool*) – use Gaussian kernel density estimates instead of histograms.
- **bins** (*int or sequence*) – histogram bins passed to numpy.histogram.
- **epsilon** (*float or False*) – pseudocount added to each histogram bin when truthy.
- **base** (*float*) – logarithm base for divergence.

**Returns:** KL(x || y), KL(y || x), and Jensen-Shannon divergence. Histogram edges are determined from x; values of y outside them are excluded.

**Return type:** `tuple[float, float, float]`

## Plotting

### `plotting.plot_mutational_paths(paths_array, v1, v2, stat_freq_norm, sequence_to_index, sims, mut_stop=False, time_stop=False, base_size=3, base_alpha=0.3, threshold=1, variable_node_size=True, node_alpha=1.0)`

Draw simulated paths in diffusion coordinates, colored by log stationary frequency.

**Parameters:**

- **paths_array** (*sequence of sequences*) – simulated paths of nucleotide sequences.
- **v1** (*array-like*) – first diffusion coordinate indexed by sequence.
- **v2** (*array-like*) – second diffusion coordinate indexed by sequence.
- **stat_freq_norm** (*np.ndarray*) – stationary frequencies indexed by sequence.
- **sequence_to_index** (*dict[str, int]*) – sequence to coordinate index.
- **sims** (*int*) – number of simulations, used in count adjustment and title.
- **mut_stop** (*int or False*) – number of mutation events; takes precedence when truthy.
- **time_stop** (*float or False*) – end time; in simulation functions, enables a time stopping rule when truthy.
- **base_size** (*float*) – baseline node size.
- **base_alpha** (*float*) – edge opacity multiplier per traversal, capped at one.
- **threshold** (*int*) – minimum visit count used for display; exact comparison depends on the plot.
- **variable_node_size** (*bool*) – scale node size with visit count.
- **node_alpha** (*float*) – node opacity.

**Returns:** creates a Matplotlib figure. Starting sequences are labeled; node counts and edge opacity reflect trajectory visits.

**Return type:** `None`

### `plotting.plot_kmer_occurrence(x, y, sequences, count_maps, colors, starting_seqs_list, labels=[], threshold=10, ax=None)`

Plot k-mer occurrences by group and highlight starting sequences.

**Parameters:**

- **x** (*array-like*) – first sample or x coordinates, according to the function.
- **y** (*array-like*) – second sample or y coordinates, according to the function.
- **sequences** (*sequence[str]*) – ordered sequences; indices must align with associated vectors or matrices.
- **count_maps** (*sequence[dict]*) – visit counts per sequence for each group.
- **colors** (*sequence*) – Matplotlib colors for groups.
- **starting_seqs_list** (*sequence of sequences*) – starting sequences per group.
- **labels** (*sequence or dict*) – group labels for occurrence plots, or amino acid label mapping for dumbbell plots.
- **threshold** (*int*) – minimum visit count used for display; exact comparison depends on the plot.
- **ax** (*matplotlib.axes.Axes or None*) – axes to draw on; None uses current axes.

**Returns:** draws on ax or the current axes. A k-mer is assigned to the first group exceeding threshold; only groups paired with labels are drawn in color.

**Return type:** `None`

### `plotting.dumbbellplot(df, cdr=None, order=None, labels=None, naive_color=color_blue, memory_color=color_violet, line_color=2, aa_col=None, naive_col=None, memory_col=None, ax=None)`

Compare naive and memory amino acid frequencies with connected points.

**Parameters:**

- **df** (*pandas.DataFrame*) – amino acid and naive/memory frequency columns.
- **cdr** (*str or None*) – CDR name used in column inference and plot title.
- **order** (*sequence or None*) – desired amino acid order.
- **labels** (*sequence or dict*) – group labels for occurrence plots, or amino acid label mapping for dumbbell plots.
- **naive_color** (*color*) – color for naive frequencies.
- **memory_color** (*color*) – color for memory frequencies.
- **line_color** (*float*) – width of connecting segments.
- **aa_col** (*str or None*) – amino acid column; inferred when omitted.
- **naive_col** (*str or None*) – naive frequency column; inferred when omitted.
- **memory_col** (*str or None*) – memory frequency column; inferred when omitted.
- **ax** (*matplotlib.axes.Axes or None*) – axes to draw on; None uses current axes.

**Returns:** axes containing the plot.

**Return type:** `matplotlib.axes.Axes`

### `plotting.add_edges(connectivity_dict, x_vector1, y_vector2, ax, linewidth=0.1, alpha=0.2, color=color_gray, zorder=0, rasterized=True)`

Add undirected connectivity edges to an existing scatter plot.

**Parameters:**

- **connectivity_dict** (*dict[int, iterable[int]]*) – index adjacency mapping.
- **x_vector1** (*array-like*) – x coordinates indexed by node.
- **y_vector2** (*array-like*) – y coordinates indexed by node.
- **ax** (*matplotlib.axes.Axes or None*) – axes to draw on; None uses current axes.
- **linewidth** (*float*) – edge width.
- **alpha** (*float*) – edge opacity.
- **color** (*color*) – edge color.
- **zorder** (*float*) – drawing order of edges.
- **rasterized** (*bool*) – rasterize the edge collection in vector output.

**Returns:** the supplied axes with a LineCollection added.

**Return type:** `matplotlib.axes.Axes`

### `plotting.normal_palette(base_color, n, sigma_h=0, sigma_s=0, sigma_v=0, seed=None)`

Generate colors by adding independent normal noise in HSV space.

**Parameters:**

- **base_color** (*color*) – Matplotlib-compatible base color.
- **n** (*int*) – number of colors.
- **sigma_h** (*float*) – standard deviation of hue noise.
- **sigma_s** (*float*) – standard deviation of saturation noise.
- **sigma_v** (*float*) – standard deviation of value noise.
- **seed** (*int or None*) – seed for the local random generator.

**Returns:** n hex colors. Hue wraps around; saturation and value are clipped to [0, 1].

**Return type:** `list[str]`

### `plotting.path_edges(seqs_array, sequence_to_index)`

Convert sequence trajectories into undirected index adjacency maps.

**Parameters:**

- **seqs_array** (*sequence of sequences*) – nucleotide trajectories aligned with times_array.
- **sequence_to_index** (*dict[str, int]*) – sequence to coordinate index.

**Returns:** one adjacency map per trajectory.

**Return type:** `list[dict[int, set[int]]]`

## Shared colors

`scripts.functions.colors.shm_colors` provides named colors for SHM figures. The plotting module also defines its own color constants.
