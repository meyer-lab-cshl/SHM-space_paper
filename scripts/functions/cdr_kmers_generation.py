import numpy as np
import tqdm
############################ k-mer generation ############################

def kmers(sequence, kmer):
    """
    Collect distinct, complete k-mers without N from a sequence.
    :param sequence: str: nucleotide sequence.
    :param kmer: int: k-mer length.
    :return: list[str]: distinct k-mers in unspecified order.
    """
    lst = []
    for i in range(len(sequence)):
        if len(sequence[i:i + kmer]) == kmer and 'N' not in sequence[i:i + kmer]:
            lst.append(sequence[i:i + kmer])
    return list(set(lst))

def kmer_gene_generation(cdrs_table, k, category, gene_id, seq_to_idx, printed = False, downsample = False, unique = False):

    """
    Collect CDR1 and CDR2 k-mers for a gene or gene class.
    :param cdrs_table: pandas.DataFrame: rows with gene_class, gene, nt_sequence and CDR columns.
    :param k: int: k-mer length or number of requested eigenvalues, according to the function.
    :param category: str: 'gene_class' selects the gene_class column; any other value selects gene.
    :param gene_id: str: gene or gene-class identifier to select.
    :param seq_to_idx: dict[str, int]: mapping from k-mers to their indices.
    :param printed: bool: whether to print unique k-mer counts.
    :param downsample: int or False: number of distinct genes to sample without replacement.
    :param unique: bool: whether to deduplicate each CDR k-mer list.
    :return: tuple: number of genes, CDR1 k-mers, CDR2 k-mers, and their respective index lists. Repeated k-mers are retained unless unique is true.
    """

    cdr1_class = []
    cdr2_class = []

    ## select genes before extracting CDR windows
    if category == 'gene_class':
        df = cdrs_table[cdrs_table['gene_class'] == gene_id].reset_index(drop = True)
    else:
        df = cdrs_table[cdrs_table['gene'] == gene_id].reset_index(drop=True)
    if downsample:
        gene_downsample = np.random.choice(list(set(df['gene'])), size = downsample, replace = False)
        df = df[df['gene'].isin(gene_downsample)].copy().reset_index(drop = True)

    ## retain complete windows without ambiguous nucleotides
    for i in tqdm.tqdm(range(len(df))):
        nt = df['nt_sequence'][i]
        cdr_type = df['CDR'][i]
        k_mers = []

        for x in range(len(nt)):
            k_mer = nt[x:x+k]
            if len(k_mer) == k and 'N' not in k_mer:
                k_mers.append(k_mer)

        if cdr_type == 'CDR1':
            cdr1_class = cdr1_class + k_mers
        elif cdr_type == 'CDR2':
            cdr2_class = cdr2_class + k_mers

    if unique:
        cdr1_class = list(set(cdr1_class))
        cdr2_class = list(set(cdr2_class))


    if printed:
        print(f'Number of unique k-mers in CDR1 of {gene_id}: {len(set(cdr1_class))}')
        print(f'Number of unique k-mers in CDR2 of {gene_id}: {len(set(cdr2_class))}')

    cdr1_idx = [seq_to_idx[i] for i in cdr1_class]
    cdr2_idx = [seq_to_idx[i] for i in cdr2_class]


    return len(set(df['gene'])), cdr1_class, cdr2_class, cdr1_idx, cdr2_idx