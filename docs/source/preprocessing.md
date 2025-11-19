# Preprocessing: `pp`

Utilities used by scBIOT to prepare AnnData objects for OT alignment and the
Transformer VAE. These mirror what you see in the tutorials.

**AnnData registration**
- `setup_anndata`

**ATAC preprocessing**
- `remove_promoter_proximal_peaks`
- `find_variable_features`
- `add_iterative_lsi`
- `annotate_gene_activity`

**Optimal transport**
- `ot.integrate`
- `ot.build_aligned_coembedding`
- `ot.label_transfer_shared_pca`


See the tutorial pages for code snippets that call each function.
