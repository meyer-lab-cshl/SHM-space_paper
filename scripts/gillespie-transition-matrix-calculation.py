import argparse

parser = argparse.ArgumentParser(description='Transition matrix calculation')
parser.add_argument('-job_id', type=str)
parser.add_argument('-nseq', type=int)
parser.add_argument('-tau_stop', type=int)
parser.add_argument('-tau_collect', type=int)
args = parser.parse_args()

import tqdm

import numpy as np
import pandas as pd

from functions import Gillespie_algorithm as ga
from functions import mutation_matrix as mm

muts = pd.read_csv('data/Spisak_mutabilities_context.tsv', sep = "\t")
## sort transition matrix
sequences = mm.generate_all_sequences(5)
muts['Motif'] = pd.Categorical(muts['Motif'], categories=sequences, ordered=True)
muts = muts.sort_values('Motif').reset_index(drop=True)
mutabilities = np.array(muts['Mutability'])
muts_array = np.array(muts)

to_mutate = pd.read_csv('data/2025-09-16_OAS/sequences_to_mutate.csv')
to_mutate = to_mutate[~(to_mutate['sequence'].str.contains('N'))].reset_index(drop = True)
to_mutate = to_mutate[['sequence']].drop_duplicates()
print(len(to_mutate))

resulting_table = np.zeros((len(sequences), 8))
resulting_table[:, 0] = list(range(len(sequences)))

nts = {'A': 2, 'C': 3, 'G': 4, 'T': 5}

k = 5

for tomut_seq in tqdm.tqdm(list(to_mutate['sequence'])[:args.nseq]):

    tomut_dict, tomut_weights = ga.kmer_positions_preseeded(tomut_seq, sequences, k, looped = False, overlapping = True)

    tau_total = 0
    tau_total_burnt = 0

    while tau_total_burnt < args.tau_stop:
        ## chose position to be mutated
        tau, chosen_mer, chosen_position = ga.which_position(tomut_dict, tomut_weights, mutabilities)

        ## calculate exposure of all contexts present during this interval (pre-event counts)
        if tau_total_burnt + tau > args.tau_collect and tau_total_burnt + tau < args.tau_stop:
            resulting_table[:, 1] += tomut_weights * tau  # pre-event counts

        ## mutation event
        chosen_nt = ga.which_mutation(chosen_mer, muts_array)
        tomut_dict, tomut_weights = ga.resulting_dict(sequences, chosen_nt, chosen_mer, chosen_position, tomut_dict, tomut_weights, len(tomut_seq))

        ## store the per-nt results
        tau_total_burnt += tau
        if tau_total_burnt > args.tau_collect:
            tau_total += tau
            resulting_table[chosen_mer, nts[chosen_nt]] += 1

    ## k-mer counts after all iterations
    resulting_table[:, 6] += tomut_weights

    filtered_dict = {i: [p for p in ps if (p - 0) % k == 0]
                 for i, ps in tomut_dict.items()}

    ## filtered weights
    filtered_counts = np.fromiter((len(ps) for ps in filtered_dict.values()),
                              dtype=np.int64, count=len(tomut_dict))

    resulting_table[:, 7] += filtered_counts

resulting_table = pd.DataFrame(resulting_table)

resulting_table['Motif'] = sequences
resulting_table.columns = ['Ind', 'Time', '# A', '# C', '# G', '# T', 'counts', 'nonoverlap counts', 'Motif']
resulting_table = resulting_table[['Motif', 'Time', '# A', '# C', '# G', '# T', 'counts', 'nonoverlap counts']]
resulting_table['# events'] = (resulting_table['# A'] + resulting_table['# C'] + resulting_table['# G'] + resulting_table['# T'])
resulting_table['Mutability'] = resulting_table['# events']/resulting_table['Time']
resulting_table['To A'] = resulting_table['# A'] /(resulting_table['# events'])
resulting_table['To C'] = resulting_table['# C'] /(resulting_table['# events'])
resulting_table['To G'] = resulting_table['# G'] /(resulting_table['# events'])
resulting_table['To T'] = resulting_table['# T'] /(resulting_table['# events'])
resulting_table['stat freq'] = resulting_table['counts'] / np.sum(resulting_table['counts'])
resulting_table['nonoverlap stat freq'] = resulting_table['nonoverlap counts'] / np.sum(resulting_table['nonoverlap counts'])

resulting_table.to_csv(f'results/transition-matrix_calculation/id{args.job_id}-nseq{args.nseq}-stop{args.tau_stop}-burnt{args.tau_collect}_mutability.csv', index=False)