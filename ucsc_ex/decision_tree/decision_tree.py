from digits_pca import get_prinicipal_features_and_labels, draw_scatter_plot
from utils_stump import build_tree, evaluate_tree, plot_contours
from commons import traverse_tree, log_debug, log
from sklearn.metrics import confusion_matrix
from config import *
import numpy as np


def main_task():
    xi, labels = get_prinicipal_features_and_labels()

    labels[labels == NEGATIVE_CLASS] = NEGATIVE_CLASS_MAPPED
    labels[labels == POSITIVE_CLASS] = POSITIVE_CLASS_MAPPED
    x_nd = np.column_stack((xi, labels))

    root_node = build_tree(x_nd)

    stats_dict = {}
    traverse_tree(root_node, stats_dict)
    log(stats_dict)

    target_actual = [0] * np.alen(x_nd)
    target_predicted = [0] * np.alen(x_nd)
    for idx in range(0, np.alen(x_nd)):
        target_actual[idx] = x_nd[idx][NUM_FEATURES]
        target_predicted[idx] = evaluate_tree((x_nd[idx][:NUM_FEATURES]), root_node)

    plot_contours(x_nd, target_actual, root_node)
    cm = confusion_matrix(target_actual, target_predicted)
    log("Accuracy: ", (cm[0][0] + cm[1][1]) / (np.sum(cm)))

if __name__ == '__main__':
    main_task()
