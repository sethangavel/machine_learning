import globals
import numpy as np
import numpy.linalg as lalg
from commons import DNode, log_debug, log
from config import *
import matplotlib.pyplot as plt


def get_feature_impurity_and_tau(x, t):
    x_nd_raw = np.column_stack(zip(x, t)).transpose()
    x_nd = x_nd_raw[x_nd_raw[:, 0].argsort(kind='mergesort')]
    t = x_nd[:, 1]

    x_count = np.alen(x_nd)
    x_negative = [x_nd[x, 0] for x in range(0, x_count) if x_nd[x, 1] == NEGATIVE_CLASS_MAPPED]
    t_negative = np.alen(x_negative)
    x_positive = [x_nd[x, 0] for x in range(0, x_count) if x_nd[x, 1] == POSITIVE_CLASS_MAPPED]
    t_positive = np.alen(x_positive)
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
        impurity_tmp = (1.0 / x_count) * (impurity_part2_1 + impurity_part2_2)
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
    assert x_t_all.shape[1] == NUM_FEATURES + 1

    num_features = NUM_FEATURES
    target_idx = NUM_FEATURES

    delta_array = np.zeros(num_features)
    tau_array = np.zeros(num_features)
    target = x_t_all[:, target_idx]
    for feature_idx in range(0, num_features):
        feature = x_t_all[:, feature_idx]
        delta, tau = get_feature_impurity_and_tau(feature, target)
        delta_array[feature_idx] = delta
        tau_array[feature_idx] = tau
        log_debug("\n")
    assert np.min(delta_array) > 0
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
    # assert prevalence_negative * prevalence_positive < LIMIT_LEAF_NODE_PREVALENCE
    return get_leaf_node_by_prevalence(prevalence_negative, prevalence_positive)


def build_tree_recursive(x_t_all, level=0):
    x_t_len = np.alen(x_t_all)
    assert np.alen(x_t_all > 0)
    if globals.tree_height < level:
        globals.tree_height = level

    index_target = NUM_FEATURES
    prevalence_negative, prevalence_positive = get_prevalence(x_t_all[:, index_target])
    prevalence = prevalence_negative * prevalence_positive

    log_debug("Tree max height so far: ", globals.tree_height)
    log("X very pure? : subset len: {}, prevalence: {}", x_t_len, prevalence)
    if prevalence < LIMIT_LEAF_NODE_PREVALENCE or x_t_len < LIMIT_LEAF_NODE_SUBSET_SIZE:
        log_debug("X very pure. Bailing out: subset len: {}, prevalence: {}", x_t_len, prevalence)
        return get_leaf_node_by_prevalence(prevalence_negative, prevalence_positive)

    delta_array, tau_array = get_delta_and_tow_impl(x_t_all)
    delta_max_idx = np.argmax(delta_array)
    tau = tau_array[delta_max_idx]
    log_debug("delta_array: ", delta_array, ", delta_max_idx: ", delta_max_idx, ", tau: ", tau_array)

    x_t_all_sorted_delta_max = x_t_all[x_t_all[:, delta_max_idx].argsort(kind='mergesort')]
    x_delta_max = x_t_all_sorted_delta_max[:, delta_max_idx]
    log_debug("Tau, idx: ", tau, np.where(x_delta_max == tau), ", x_sorted: ", x_delta_max)
    tau_idx = np.where(x_delta_max == tau)[0][0]
    assert (tau_idx >= 0) and (tau_idx <= np.alen(x_t_all_sorted_delta_max))
    # log_debug("\n level: ", level, ", tau_idx: ", tau_idx)

    x_t_all_left = x_t_all_sorted_delta_max[0:tau_idx, :]
    x_t_all_right = x_t_all_sorted_delta_max[tau_idx:, :]
    if np.alen(x_t_all_left) > 0 and np.alen(x_t_all_right) > 0:
        node = DNode("RULE", feature_idx=delta_max_idx, tau=tau)

        assert (tau_idx > 0)
        # log_debug("\n level:", level, ", x_left: ", x_t_all_left.shape[0])
        node.left = build_tree_recursive(x_t_all_left, level + 1)

        assert(np.alen(x_t_all_sorted_delta_max) - tau_idx > 0)
        # log_debug("\n level: ", level, ", x_right: ", x_t_all_right.shape[0])
        node.right = build_tree_recursive(x_t_all_right, level + 1)
    else:
        assert np.alen(x_t_all_left) == 0 or np.alen(x_t_all_right) == 0
        node = get_leaf_node(x_t_all_sorted_delta_max[:, 2])

    return node


def build_tree(x_t_all):
    return build_tree_recursive(x_t_all)


def evaluate_tree(xi, root_node):

    if root_node.is_leaf():
        return root_node.target

    decision_feature_idx = root_node.feature_index
    decision_feature_tau = root_node.tau
    if xi[decision_feature_idx] < decision_feature_tau:
        return evaluate_tree(xi, root_node.left)
    else:
        return evaluate_tree(xi, root_node.right)


def plot_contours(x_nd, y, root_node):
    import matplotlib.pyplot as plt
    import numpy as np

    X = x_nd[:, [0, 1]]

    fig = plt.figure()
    cols = np.zeros((np.alen(y), 4))
    for idx, ll in enumerate(y):
        if ll == POSITIVE_CLASS_MAPPED:
            cols[idx] = [1, 0, 0, SCATTER_PLOT_ALPHA]
        if ll == NEGATIVE_CLASS_MAPPED:
            cols[idx] = [0, 0.2, 1, SCATTER_PLOT_ALPHA]
    random_order = np.arange(np.alen(y))
    ax = fig.add_subplot(111, facecolor='white')
    ax.set_aspect('equal')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, PLOT_STEP),
                         np.arange(y_min, y_max, PLOT_STEP))

    z = np.zeros(shape=xx.shape)
    for row in range(0, xx.shape[0]):
        for col in range(0, xx.shape[1]):
            z[row][col] = evaluate_tree((xx[row][col], yy[row][col]), root_node)

    z = z.reshape(xx.shape)
    cs = ax.contourf(yy, xx, z, alpha=DECISION_BOUNDARY_ALPHA, cmap=plt.cm.Paired)

    ax.scatter(x_nd[:, 1], x_nd[:, 0], s=8, linewidths=0,
               facecolors=cols[random_order, :], marker='o')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("tight")

    """
    idx = np.where(y == NEGATIVE_CLASS_MAPPED)
    plt.scatter(X[idx, 1], X[idx, 0], c='b', cmap=plt.cm.Paired)
    idx = np.where(y == POSITIVE_CLASS_MAPPED)
    plt.scatter(X[idx, 1], X[idx, 0], c='r', cmap=plt.cm.Paired)
    plt.gca().invert_yaxis()
    """

    plt.rcParams['axes.facecolor'] = 'b'
    # plt.gca().invert_yaxis()
    plt.show()





