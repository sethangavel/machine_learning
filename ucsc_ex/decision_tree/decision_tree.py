from digits_pca import get_prinicipal_features_and_labels, draw_scatter_plot
from utils_stump import build_tree, evaluate_tree
from commons import traverse_tree, log_debug, log
from sklearn.metrics import confusion_matrix
from config import *
import numpy as np


def main_task():
    x1, x2, x3, labels = get_prinicipal_features_and_labels()
    if ENABLE_PLOT:
        draw_scatter_plot(x1, x2, (0, 0), (0, 1), labels)
    x1 = np.array([x1]).transpose()
    x2 = np.array([x2]).transpose()
    x3 = np.array([x3]).transpose()

    log_debug("X1: ", x1.shape)
    log_debug("X2: ", x2.shape)
    log_debug("X3: ", x3.shape)
    log_debug("T: ", labels.shape)

    labels[labels == NEGATIVE_CLASS] = NEGATIVE_CLASS_MAPPED
    labels[labels == POSITIVE_CLASS] = POSITIVE_CLASS_MAPPED
    x_nd = np.column_stack((x1, x2, x3, labels))

    root_node = build_tree(x_nd)

    stats_dict = {
        'T{}'.format(NEGATIVE_CLASS_MAPPED): 0,
        'T{}'.format(POSITIVE_CLASS_MAPPED): 0,
        'F0': 0,
        'F1': 0,
        'F2': 0,
    }
    traverse_tree(root_node, stats_dict)
    log(stats_dict)

    target_actual = [0] * np.alen(x_nd)
    target_predicted = [0] * np.alen(x_nd)
    for idx in range(0, np.alen(x_nd)):
        target_actual[idx] = x_nd[idx][NUM_FEATURES]
        target_predicted[idx] = evaluate_tree((x_nd[idx][:NUM_FEATURES]), root_node)

    cm = confusion_matrix(target_actual, target_predicted)
    log("Accuracy: ", (cm[0][0] + cm[1][1]) / (np.sum(cm)))

if __name__ == '__main__':
    main_task()
