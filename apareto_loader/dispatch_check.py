"""Independent CPU oracle for pre-mask routing versus actual kernel dispatch."""
import numpy as np


def rescue_terms(ids, weights, keep, x, cap, gain=None):
    """The rescue rule's oracle for one (token, layer) row: the up to `cap` terms the keep-set
    drops whose score is above x, highest first, the first term on a tie. The score is the routing weight, or with
    `gain` (this layer's [experts] row) the fp32 product weight x gain[expert], as the ring kernel computes it."""
    def score(k):
        w = np.float32(weights[k])
        return w if gain is None else np.float32(w * np.float32(gain[int(ids[k])]))
    dropped = [k for k in range(len(ids)) if not keep[int(ids[k])] and score(k) > np.float32(x)]
    return set(sorted(dropped, key=lambda k: (-float(score(k)), k))[:cap])


def verify_dispatch(original_ids, original_weights, actual_slots, actual_weights,
                    intended_keep, slot_to_seat, *, served_keep=None, rescue=None):
    """Check original top-k terms against independent kernel-input captures.

    No serving remap helper is imported. Routes: [tokens,layers,top_k]; keep:
    [layers,experts]; inverse physical map: [layers,capacity], -1 if unowned.
    rescue: {'x', 'cap' (int, or one per layer)[, 'gain': [layers, experts]]} when the rescue rule was live for these rows: exactly the oracle's dropped terms
    (rescue_terms) are served, by their own expert with their routing weight; every other dropped term by nothing.
    """
    original_ids, actual_slots = np.asarray(original_ids), np.asarray(actual_slots)
    original_weights, actual_weights = np.asarray(original_weights), np.asarray(actual_weights)
    intended_keep, slot_to_seat = np.asarray(intended_keep), np.asarray(slot_to_seat)
    shape = original_ids.shape
    if (len(shape) != 3 or min(shape) < 1 or original_ids.dtype.kind not in "iu" or
            actual_slots.shape != shape or actual_slots.dtype.kind != "i" or
            original_weights.shape != shape or actual_weights.shape != shape or
            original_weights.dtype.kind != "f" or actual_weights.dtype != original_weights.dtype or
            intended_keep.ndim != 2 or intended_keep.dtype != np.bool_ or intended_keep.shape[0] != shape[1] or
            slot_to_seat.ndim != 2 or slot_to_seat.dtype.kind != "i" or slot_to_seat.shape[0] != shape[1]):
        raise ValueError("Malformed independent dispatch captures")
    if (not np.isfinite(original_weights).all() or not np.isfinite(actual_weights).all() or
            np.any(original_ids < 0) or np.any(original_ids >= intended_keep.shape[1]) or
            np.any(actual_slots < -1) or np.any(actual_slots >= slot_to_seat.shape[1])):
        raise ValueError("Invalid routing IDs, slots or weights")
    if np.any(np.diff(np.sort(original_ids, axis=-1), axis=-1) == 0):
        raise ValueError("Duplicate original picks or missing router capture")
    if served_keep is not None and not np.array_equal(np.asarray(served_keep), intended_keep):
        raise ValueError("Served mask differs from the request's intended keep-set")
    expected_drops, survivors, empty_rows, rescued = 0, 0, 0, 0
    for token in range(shape[0]):
        for layer in range(shape[1]):
            row_survivors = 0
            chosen = (set() if rescue is None else
                      rescue_terms(original_ids[token, layer], original_weights[token, layer], intended_keep[layer],
                                   rescue['x'], rescue['cap'][layer] if isinstance(rescue['cap'], (list, tuple)) else rescue['cap'],
                                   None if rescue.get('gain') is None else rescue['gain'][layer]))
            for term in range(shape[2]):
                expert = int(original_ids[token, layer, term])
                slot = int(actual_slots[token, layer, term])
                if intended_keep[layer, expert]:
                    if slot < 0 or int(slot_to_seat[layer, slot]) != expert:
                        raise ValueError(f"Allowed pick dropped or mapped to wrong weights at {token}/{layer}/{term}")
                    if actual_weights[token, layer, term].tobytes() != original_weights[token, layer, term].tobytes():
                        raise ValueError(f"Allowed routing weight changed at {token}/{layer}/{term}")
                    row_survivors += 1
                elif term in chosen:
                    if slot < 0 or int(slot_to_seat[layer, slot]) != expert:
                        raise ValueError(f"Rescued pick not served by its expert at {token}/{layer}/{term}")
                    if actual_weights[token, layer, term].tobytes() != original_weights[token, layer, term].tobytes():
                        raise ValueError(f"Rescued routing weight changed at {token}/{layer}/{term}")
                    rescued += 1
                elif slot != -1 or actual_weights[token, layer, term] != 0:
                    raise ValueError(f"Disallowed pick was dispatched at {token}/{layer}/{term}")
                else:
                    expected_drops += 1
            survivors += row_survivors
            empty_rows += row_survivors == 0
    return {"clean": True, "forwarded_tokens": shape[0], "layers": shape[1],
        "surviving_terms": survivors, "dropped_terms": expected_drops,
        "all_dropped_rows": empty_rows, "exact_ids_and_weights": True,
        **({} if rescue is None else {"rescued_terms": rescued,
                                      "rescue": {k: v for k, v in rescue.items() if k != 'gain'} |
                                      {"gain": rescue.get('gain') is not None}})}
