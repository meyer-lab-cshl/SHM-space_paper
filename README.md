# Dynamical consequences of context-dependent mutation: visualizing antibody sequence evolution during somatic hypermutation

Supplementary repository to the paper: [bioRxiv version](https://www.biorxiv.org/content/10.64898/2026.10.26.762018v1).

Context-dependent mutation models describe how local sequence context affects individual substitution rates, but how these biases shape sequence evolution over multiple mutational steps remains challenging to characterize. Here, we develop a framework for analyzing and visualizing long-term sequence dynamics arising from context-dependent, non-reversible mutation processes. We apply it to somatic hypermutation during antibody affinity maturation, using previously inferred context-dependent mutation rates. Our analysis recovers conventional hypermutation hot- and cold-spot patterns while exposing substantial heterogeneity in mutability within these categories. We also identify sequences with low mutability that lack conventional cold-spot motifs. Applying the framework to antigen-binding regions from different antibody-encoding gene families reveals how differences in sequence composition lead to distinct mutational trajectories through sequence space, suggesting that hyper-mutation biases constrain how these regions diversify. We further show how mutational biases defined at the level of 5-nucleotide contexts propagate to the codon and amino acid levels, producing low stationary frequencies for stop codons and tyrosine and high stationary frequencies for proline. Changes in amino acid composition between naive and memory B cells are qualitatively consistent with these predictions. Together, our results provide a general framework for connecting local context-dependent mutation rates to long-term sequence evolution and demonstrate how intrinsic somatic hypermutation biases shape the relative accessibility of different regions of sequence space during affinity maturation.

### Figures

Every figure has a separate notebook:
- Main text figures (Main_)
- Extended data figures (Extended-Data_)

### Code

Custom functions necessary for the figure making and conducted analyses are in the "scripts/functions" folder, with their documentation in "function-reference.html" file.

Code for the Gillespie simulations on BCR sequences is in "scripts/gillespie-transition-matrix-claculation.py" file.

All necessary packages and their versions are in the "environment.yml" file.


### Authors

Vasilisa A. Kovaleva, Hannah V. Meyer, David M. McCandlish