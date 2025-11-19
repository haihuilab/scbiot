# Optimal transport: `ot`

OT utilities for aligning batches and modalities. The functions below match what
you see in the tutorials; refer to the notebooks for full, runnable examples.

- `integrate`: batch correction for single-modality data (RNA or ATAC).
- `integrate_ot`: OT-only integration when you want to supply your own embedding.
- `integrate_paired`: joint integration for paired multiome inputs (RNA + ATAC).
- `build_aligned_coembedding`: create a shared embedding for unpaired modalities.
- `harmonize_gene_names`: ensure gene naming across RNA/ATAC inputs matches.
- `label_transfer_shared_pca`: project labels across modalities in a shared PCA space.
