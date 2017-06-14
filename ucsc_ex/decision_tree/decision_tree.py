from digits_pca import get_training_prinicipal_features_and_labels, get_test_prinicipal_features_and_labels
from utils_stump import build_tree, evaluate_tree, plot_contours
from commons import traverse_tree, log_debug, log
from sklearn.metrics import confusion_matrix
from config import *
import numpy as np


def main_task():
    # Training
    xi, labels = get_training_prinicipal_features_and_labels()
    labels[labels == NEGATIVE_CLASS] = NEGATIVE_CLASS_MAPPED
    labels[labels == POSITIVE_CLASS] = POSITIVE_CLASS_MAPPED
    x_nd = np.column_stack((xi, labels))
    root_node = build_tree(x_nd)
    stats_dict = {}
    traverse_tree(root_node, stats_dict)
    log(stats_dict)
    training_target_actual = [0] * np.alen(x_nd)
    for idx in range(0, np.alen(x_nd)):
        training_target_actual[idx] = x_nd[idx][NUM_FEATURES]
    plot_contours(x_nd, training_target_actual, root_node)

    # Testing
    test_xi, test_labels = get_test_prinicipal_features_and_labels()
    test_labels[test_labels == NEGATIVE_CLASS] = NEGATIVE_CLASS_MAPPED
    test_labels[test_labels == POSITIVE_CLASS] = POSITIVE_CLASS_MAPPED
    test_x_nd = np.column_stack((test_xi, test_labels))
    test_target_actual = [0] * np.alen(test_x_nd)
    test_target_predicted = [0] * np.alen(test_x_nd)
    for idx in range(0, np.alen(test_x_nd)):
        test_target_actual[idx] = x_nd[idx][NUM_FEATURES]
        test_target_predicted[idx] = evaluate_tree((test_x_nd[idx][:NUM_FEATURES]), root_node)

    plot_contours(test_x_nd, test_target_actual, root_node)
    cm = confusion_matrix(test_target_actual, test_target_predicted)
    log("Accuracy: ", (cm[0][0] + cm[1][1]) / (np.sum(cm)))

if __name__ == '__main__':
    main_task()
