from __future__ import annotations

from typing import Any, Optional, Tuple


def _resolve_reference_query(
    obs,
    reference: Optional[Any] = None,
    query: Optional[Any] = None,
) -> Tuple[Any, Any, Any, Any]:
    values = list(obs.unique())
    if reference is not None or query is not None:
        if reference is None or query is None:
            raise ValueError("Provide both reference and query batch labels.")
        if reference not in values or query not in values:
            raise ValueError("reference/query labels not found in batch_key column.")
        ref_label, query_label = reference, query
    elif "reference" in values and "query" in values:
        ref_label, query_label = "reference", "query"
    elif len(values) == 2:
        ref_label, query_label = values[0], values[1]
    else:
        raise ValueError("Could not infer reference/query batches; pass reference and query.")

    ref_mask = obs == ref_label
    query_mask = obs == query_label
    if int(ref_mask.sum()) == 0 or int(query_mask.sum()) == 0:
        raise ValueError("Reference/query batch masks are empty.")
    return ref_mask, query_mask, ref_label, query_label
