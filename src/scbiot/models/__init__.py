# scbiot/models/__init__.py
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .vae import VAE
from .vae_train import (
    BATCH_SIZE,
    D_TOKEN,
    FACTOR,
    LR,
    N_HEAD,
    NUM_EPOCHS,
    NUM_LAYERS,
    HyperParams,
    VAEModel,
)

if TYPE_CHECKING:
    from anndata import AnnData


def _is_anndata(obj: object) -> bool:
    try:
        from anndata import AnnData  # type: ignore
    except ImportError:
        return False
    return isinstance(obj, AnnData)


def vae(
    adata: AnnData | int,
    num_layers: int | None = NUM_LAYERS,
    categories: Sequence[int] | None = None,
    d_token: int | None = D_TOKEN,
    *,
    n_head: int = N_HEAD,
    factor: int = FACTOR,
    bias: bool = True,
    var_key: str | None = None,
    batch_key: str | None = None,
    pseudo_key: str | None = None,
    true_key: str | None = None,
    threshold: float = 0.1,
    num_clusters: int = 64,
    random_seed: int = 42,
    training_steps: int | None = None,
    device: str | None = None,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    num_epochs: int = NUM_EPOCHS,
    hyperparams: HyperParams | None = None,
    prior_pcr: float = 1.0,
    verbose: bool = False,
    _raw: bool | None = None,
    d_numerical: int | None = None,
    hid_dim: int | None = None,
    num_batches: int = 1,
) -> VAE | VAEModel:
    """
    Factory that returns the high-level training wrapper by default.

    Pass `_raw=True` to receive the bare torch.nn.Module implementation and
    provide `d_numerical` + `hid_dim` (or an integer in `adata`).

    Examples
    --------
    Basic usage:

    >>> import scbiot as scb
    >>> model = scb.models.vae(
    ...     adata,
    ...     num_layers=2,
    ...     var_key="scBIOT_OT",
    ...     batch_key="batch",
    ... )
    """
    raw = _raw
    if raw is None:
        raw = not _is_anndata(adata)
    if raw:
        if _is_anndata(adata):
            raise TypeError("Received AnnData with _raw=True; pass a matrix size instead.")
        if d_numerical is None:
            if isinstance(adata, int):
                d_numerical = adata
            else:
                raise TypeError("d_numerical must be provided when _raw=True.")
        if num_layers is None:
            raise TypeError("num_layers must be provided when _raw=True.")
        if hid_dim is None:
            raise TypeError("hid_dim must be provided when _raw=True.")
        return VAE(
            d_numerical,
            categories,
            num_layers,
            hid_dim,
            n_head=n_head,
            factor=factor,
            bias=bias,
            num_batches=num_batches,
        )
    if not _is_anndata(adata):
        raise TypeError("vae expects an AnnData object when _raw is False.")
    return VAEModel(
        adata,
        num_layers=num_layers,
        categories=categories,
        d_token=d_token,
        n_head=n_head,
        factor=factor,
        bias=bias,
        var_key=var_key,
        batch_key=batch_key,
        pseudo_key=pseudo_key,
        true_key=true_key,
        threshold=threshold,
        num_clusters=num_clusters,
        random_seed=random_seed,
        training_steps=training_steps,
        device=device,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        hyperparams=hyperparams,
        prior_pcr=prior_pcr,
        verbose=verbose,
    )


__all__ = ["VAE", "VAEModel", "vae"]
