import globals
import numpy as np
import numpy.linalg as lalg
from commons import DNode, log_debug
from config import *


def get_feature_impurity_and_tau(x, t):
    x_nd_raw = np.column_stack(zip(x, t)).transpose()
    x_nd = x_nd_raw[x_nd_raw[:, 0].argsort(kind='mergesort')]
    # log_debug(x_nd)

    x_count = len(x_nd)
    x_negative = [x_nd[x, 0] for x in range(0, x_count) if x_nd[x, 1] == NEGATIVE_CLASS_MAPPED]
    t_negative = len(x_negative)
    x_positive = [x_nd[x, 0] for x in range(0, x_count) if x_nd[x, 1] == POSITIVE_CLASS_MAPPED]
    t_positive = len(x_positive)
    log_debug(x_count, t_negative, t_positive)

    a_negative = 0
    a_positive = 0
    impurity_initial = impurity_optimal = (t_negative * t_positive) / (x_count * x_count)
    tow_idx = 0
    tow = x[tow_idx]

    for idx in range(1, x_count):
        if t[idx - 1] == NEGATIVE_CLASS_MAPPED:
            a_negative += 1
        else:
            a_positive += 1

        impurity_part2_1 = ((a_negative * a_positive) / (a_negative + a_positive))
        impurity_part2_2 = ((t_negative - a_negative) * (t_positive - a_positive)) / (
            t_negative + t_positive - a_negative - a_positive)
        impurity_tmp = (1 / x_count) * (impurity_part2_1 + impurity_part2_2)
        if impurity_tmp < impurity_optimal:
            impurity_optimal = impurity_tmp
            tow = x[idx]
            tow_idx = idx

    delta = impurity_initial - impurity_optimal
    tau = x[tow_idx]
    log_debug("Io: ", impurity_initial)
    log_debug("Iopt: ", impurity_optimal)
    log_debug("I delta: ", delta)
    log_debug("Tow: ", tow, ", (i={})".format(tow_idx), tau)
    return delta, tau


def get_delta_and_tow_impl(x_t_all):
    assert isinstance(x_t_all, np.ndarray)
    x_t_all_dimension = x_t_all.shape
    # log_debug("x_t_all shape: ", x_t_all.shape)
    target_idx = x_t_all_dimension[1] - 1
    num_features = x_t_all_dimension[1] - 1
    delta_array = np.zeros(num_features)
    tau_array = np.zeros(num_features)
    for feature_idx in range(0, num_features):
        feature = x_t_all[:, feature_idx]
        # log_debug("F: ", feature)
        target = x_t_all[:, target_idx]
        delta, tau = get_feature_impurity_and_tau(feature, target)
        delta_array[feature_idx] = delta
        tau_array[feature_idx] = tau
        log_debug("\n")
    # log_debug("\nD:", delta_array)
    # log_debug("\nTau:", tau_array)
    return delta_array, tau_array


def compute_weight_vector(xa, target):
    xa_pseudo_inv = lalg.pinv(xa)
    w = np.dot(xa_pseudo_inv, target)
    return w


def get_linear_classifier_weights(x_all, t_all):
    num_rows = len(x_all)
    num_cols = 2
    all_features = np.zeros(shape=(num_rows, num_cols))
    for idx in range(0, num_rows):
        row = [x_all[idx]]
        row.insert(0, 1)
        all_features[idx] = row
    w = compute_weight_vector(all_features, t_all)
    log_debug(w)


def get_prevalence(target):
    prevalence_negative = np.alen(target[target == NEGATIVE_CLASS_MAPPED]) / (np.alen(target) * 1.0)
    prevalence_positive = np.alen(target[target == POSITIVE_CLASS_MAPPED]) / (np.alen(target) * 1.0)
    return prevalence_negative, prevalence_positive


def get_leaf_node_by_prevalence(prevalence_negative, prevalence_positive):
    if prevalence_negative > prevalence_positive:
        return DNode("LEAF", target=NEGATIVE_CLASS_MAPPED)
    else:
        return DNode("LEAF", target=POSITIVE_CLASS_MAPPED)


def get_leaf_node(target):
    prevalence_negative, prevalence_positive = get_prevalence(target)
    return get_leaf_node_by_prevalence(prevalence_negative, prevalence_positive)


def get_delta_and_tow(x_t_all, level=0):
    x_t_len = np.alen(x_t_all)
    assert np.alen(x_t_all > 0)
    if globals.tree_height < level:
        globals.tree_height = level

    prevalence_negative, prevalence_positive = get_prevalence(x_t_all[:, 2])
    prevalence = prevalence_negative * prevalence_positive

    log_debug("Tree max height so far: ", globals.tree_height)
    if prevalence < LIMIT_LEAF_NODE_PREVALENCE or x_t_len < LIMIT_LEAF_NODE_SUBSET_SIZE:
        log_debug("X very pure. Bailing out!")
        return get_leaf_node_by_prevalence(prevalence_negative, prevalence_positive)

    delta_array, tau_array = get_delta_and_tow_impl(x_t_all)
    delta_max_idx = np.argmax(delta_array)
    tau = tau_array[delta_max_idx]

    x_t_all_new = x_t_all[x_t_all[:, delta_max_idx].argsort(kind='mergesort')]
    x_delta = x_t_all_new[:, delta_max_idx]
    tau_idx = np.where(x_delta == tau)[0][0]
    # log_debug("\n level: ", level, ", tau_idx: ", tau_idx)

    x_t_all_left = x_t_all_new[0:tau_idx - 1, :]
    x_t_all_right = x_t_all_new[tau_idx:, :]
    if np.alen(x_t_all_left) > 0 and np.alen(x_t_all_right) > 0:
        node = DNode("RULE", feature_idx=delta_max_idx, tau=tau)

        assert (tau_idx - 1 > 0)
        # log_debug("\n level:", level, ", x_left: ", x_t_all_left.shape[0])
        node.left = get_delta_and_tow(x_t_all_left, level + 1)

        assert(np.alen(x_t_all_new) - tau_idx > 0)
        # log_debug("\n level: ", level, ", x_right: ", x_t_all_right.shape[0])
        node.right = get_delta_and_tow(x_t_all_right, level + 1)
    else:
        assert np.alen(x_t_all_left) == 0 or np.alen(x_t_all_right) == 0
        node = get_leaf_node(x_t_all_new[:, 2])

    return node




