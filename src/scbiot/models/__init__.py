# scbiot/models/__init__.py

from .vae import VAE
from .vae_train import VAEModel


def _is_anndata(obj) -> bool:
    try:
        from anndata import AnnData  # type: ignore
    except ImportError:
        return False
    return isinstance(obj, AnnData)


def vae(*args, **kwargs):
    """
    Factory that returns the high-level training wrapper by default.

    Pass `_raw=True` to receive the bare torch.nn.Module implementation.
    """
    raw = kwargs.pop("_raw", None)
    if raw is None and args:
        raw = not _is_anndata(args[0])
    if raw:
        return VAE(*args, **kwargs)
    return VAEModel(*args, **kwargs)


__all__ = ["VAE", "VAEModel", "vae"]
