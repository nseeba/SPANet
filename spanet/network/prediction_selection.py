from typing import List, Optional
import numpy as np

import numba
from numba import njit

TArray = np.ndarray

TFloat32 = numba.types.float32
TInt64 = numba.types.int64

TPrediction = numba.typed.typedlist.ListType(TFloat32[::1])
TPredictions = numba.typed.typedlist.ListType(TFloat32[:, ::1])

TResult = TInt64[:, ::1]
TResults = TInt64[:, :, ::1]

NUMBA_DEBUG = False


if NUMBA_DEBUG:
    def njit(*args, **kwargs):
        def wrapper(function):
            return function
        return wrapper


@njit("void(float32[::1], int64, int64, float32)")
def mask_1(data, size, index, value):
    data[index] = value


@njit("void(float32[::1], int64, int64, float32)")
def mask_2(flat_data, size, index, value):
    data = flat_data.reshape((size, size))
    data[index, :] = value
    data[:, index] = value


@njit("void(float32[::1], int64, float32)")
def mask_diagonal_2(flat_data, size, value):
    data = flat_data.reshape((size, size))
    for index in range(size):
        data[index, index] = value


@njit("void(float32[::1], int64, int64, float32)")
def mask_3(flat_data, size, index, value):
    data = flat_data.reshape((size, size, size))
    data[index, :, :] = value
    data[:, index, :] = value
    data[:, :, index] = value


# @njit("void(float32[::1], int64, int64, float32)")
# def mask_4(flat_data, size, index, value):
#     data = flat_data.reshape((size, size, size, size))
#     data[index, :, :, :] = value
#     data[:, index, :, :] = value
#     data[:, :, index, :] = value
#     data[:, :, :, index] = value


# @njit("void(float32[::1], int64, int64, float32)")
# def mask_5(flat_data, size, index, value):
#     data = flat_data.reshape((size, size, size, size, size))
#     data[index, :, :, :, :] = value
#     data[:, index, :, :, :] = value
#     data[:, :, index, :, :] = value
#     data[:, :, :, index, :] = value
#     data[:, :, :, :, index] = value
#
#
# @njit("void(float32[::1], int64, int64, float32)")
# def mask_6(flat_data, size, index, value):
#     data = flat_data.reshape((size, size, size, size, size, size))
#     data[index, :, :, :, :, :] = value
#     data[:, index, :, :, :, :] = value
#     data[:, :, index, :, :, :] = value
#     data[:, :, :, index, :, :] = value
#     data[:, :, :, :, index, :] = value
#     data[:, :, :, :, :, index] = value


# @njit("void(float32[::1], int64, int64, float32)")
# def mask_7(flat_data, size, index, value):
#     data = flat_data.reshape((size, size, size, size, size, size, size))
#     data[index, :, :, :, :, :, :] = value
#     data[:, index, :, :, :, :, :] = value
#     data[:, :, index, :, :, :, :] = value
#     data[:, :, :, index, :, :, :] = value
#     data[:, :, :, :, index, :, :] = value
#     data[:, :, :, :, :, index, :] = value
#     data[:, :, :, :, :, :, index] = value
#
#
# @njit("void(float32[::1], int64, int64, float32)")
# def mask_8(flat_data, size, index, value):
#     data = flat_data.reshape((size, size, size, size, size, size, size, size))
#     data[index, :, :, :, :, :, :, :] = value
#     data[:, index, :, :, :, :, :, :] = value
#     data[:, :, index, :, :, :, :, :] = value
#     data[:, :, :, index, :, :, :, :] = value
#     data[:, :, :, :, index, :, :, :] = value
#     data[:, :, :, :, :, index, :, :] = value
#     data[:, :, :, :, :, :, index, :] = value
#     data[:, :, :, :, :, :, :, index] = value


@njit("void(float32[::1], int64, int64, int64, float32)")
def mask_jet(data, num_partons, max_jets, index, value):
    if num_partons == 1:
        mask_1(data, max_jets, index, value)
    elif num_partons == 2:
        mask_2(data, max_jets, index, value)
    elif num_partons == 3:
        mask_3(data, max_jets, index, value)
    # elif num_partons == 4:
    #     mask_4(data, max_jets, index, value)
    # elif num_partons == 5:
    #     mask_5(data, max_jets, index, value)
    # elif num_partons == 6:
    #     mask_6(data, max_jets, index, value)
    # elif num_partons == 7:
    #     mask_7(data, max_jets, index, value)
    # elif num_partons == 8:
    #     mask_8(data, max_jets, index, value)


@njit("int64[::1](int64, int64)")
def compute_strides(num_partons, max_jets):
    strides = np.zeros(num_partons, dtype=np.int64)
    strides[-1] = 1
    for i in range(num_partons - 2, -1, -1):
        strides[i] = strides[i + 1] * max_jets

    return strides


@njit(TInt64[::1](TInt64, TInt64[::1]))
def unravel_index(index, strides):
    num_partons = strides.shape[0]
    result = np.zeros(num_partons, dtype=np.int64)

    remainder = index
    for i in range(num_partons):
        result[i] = remainder // strides[i]
        remainder %= strides[i]
    return result


@njit(TInt64(TInt64[::1], TInt64[::1]))
def ravel_index(index, strides):
    return (index * strides).sum()


@njit(numba.types.Tuple((TInt64, TInt64, TFloat32))(TPrediction, TFloat32[::1]))
def maximal_prediction(predictions, offsets):
    # The branch whose best assignment has the highest log-probability plus its offset wins.
    # Returns the raw log-probability of the winning assignment.
    best_jet = -1
    best_prediction = -1
    best_score = -np.float32(np.inf)
    best_value = -np.float32(np.inf)

    for i in range(len(predictions)):
        max_jet = np.argmax(predictions[i])
        max_value = predictions[i][max_jet]
        max_score = max_value + offsets[i]

        if np.isfinite(max_value) and max_score > best_score:
            best_prediction = i
            best_score = max_score
            best_value = max_value
            best_jet = max_jet

    return best_jet, best_prediction, best_value


@njit(TResult(TPrediction, TInt64[::1], TInt64, TInt64[::1], TFloat32[::1]))
def extract_prediction(predictions, num_partons, max_jets, decoding_order, offsets):
    float_negative_inf = -np.float32(np.inf)
    max_partons = num_partons.max()
    num_targets = len(predictions)

    # Create copies of predictions for safety and calculate the output shapes
    strides = []
    for i in range(num_targets):
        strides.append(compute_strides(num_partons[i], max_jets))

    # Fill up the prediction matrix
    # -2 : Not yet assigned
    # -1 : Masked value
    # else : The actual index value
    results = np.zeros((num_targets, max_partons), np.int64) - 2

    # A physical two-jet assignment must contain two distinct jets.  Mask
    # diagonal entries before the first argmax; post-selection masking alone
    # cannot prevent (j, j) from being selected as the first pair.
    for i in range(num_targets):
        if num_partons[i] == 2:
            mask_diagonal_2(predictions[i], max_jets, float_negative_inf)

    # decoding_order[0] < 0 : greedy, the remaining branch with the highest best-assignment
    #                         log-probability plus offset chooses next.
    # otherwise           : branches choose in the fixed order given by decoding_order.
    ordered = decoding_order[0] >= 0

    for step in range(num_targets):
        if ordered:
            best_prediction = decoding_order[step]
            best_jet = np.argmax(predictions[best_prediction])
            best_value = predictions[best_prediction][best_jet]

            # Not enough unused jets left for this branch, leave it unassigned (-2).
            if not np.isfinite(best_value):
                continue
        else:
            best_jet, best_prediction, best_value = maximal_prediction(predictions, offsets)

            if not np.isfinite(best_value):
                return results

        best_jets = unravel_index(best_jet, strides[best_prediction])

        results[best_prediction, :] = -1
        for i in range(num_partons[best_prediction]):
            results[best_prediction, i] = best_jets[i]

        predictions[best_prediction][:] = float_negative_inf
        for i in range(num_targets):
            for jet in best_jets:
                mask_jet(predictions[i], num_partons[i], max_jets, jet, float_negative_inf)

    return results


@njit(TResults(TPredictions, TInt64[::1], TInt64, TInt64, TInt64[::1], numba.types.float32[:, ::1]), parallel=True)
def _extract_predictions(predictions, num_partons, max_jets, batch_size, decoding_order, offsets):
    output = np.zeros((batch_size, len(predictions), num_partons.max()), np.int64)
    predictions = [p.copy() for p in predictions]

    for batch in numba.prange(batch_size):
        current_prediction = numba.typed.List([prediction[batch] for prediction in predictions])
        output[batch, :, :] = extract_prediction(
            current_prediction, num_partons, max_jets, decoding_order, offsets[batch]
        )

    return np.ascontiguousarray(output.transpose((1, 0, 2)))


def extract_predictions(
    predictions: List[TArray],
    decoding_order: Optional[List[int]] = None,
    branch_offsets: Optional[TArray] = None,
):
    """ Choose non-overlapping jet assignments for every branch.

    Parameters
    ----------
    predictions : list of [B, T, T, ...] log-probability arrays, one per branch.
    decoding_order : optional list of branch indices. If given, every branch is listed exactly
        once and the branches choose their jets in this order. If None, SPANet's greedy
        decoding is used and the most confident remaining branch chooses next.
    branch_offsets : optional [B, NUM_BRANCHES] array added to every branch's best
        log-probability when greedy decoding compares branches, e.g. the log detection
        probability plus a fixed per-branch bias. Ignored with a fixed decoding_order.
    """
    flat_predictions = numba.typed.List([p.reshape((p.shape[0], -1)) for p in predictions])
    num_partons = np.array([len(p.shape) - 1 for p in predictions])
    max_jets = max(max(p.shape[1:]) for p in predictions)
    batch_size = max(p.shape[0] for p in predictions)

    if decoding_order is None or len(decoding_order) == 0:
        order = np.full(len(predictions), -1, dtype=np.int64)
    else:
        order = np.asarray(decoding_order, dtype=np.int64)
        if sorted(order.tolist()) != list(range(len(predictions))):
            raise ValueError(f"decoding_order must list every branch index exactly once, got {decoding_order}")

    if branch_offsets is None:
        offsets = np.zeros((batch_size, len(predictions)), dtype=np.float32)
    else:
        offsets = np.ascontiguousarray(branch_offsets, dtype=np.float32)
        if offsets.shape != (batch_size, len(predictions)):
            raise ValueError(f"branch_offsets must have shape {(batch_size, len(predictions))}, got {offsets.shape}")

    results = _extract_predictions(flat_predictions, num_partons, max_jets, batch_size, order, offsets)
    return [result[:, :partons] for result, partons in zip(results, num_partons)]
