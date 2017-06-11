from digits_pca import get_prinicipal_features_and_labels, draw_scatter_plot
from utils_stump import get_delta_and_tow
from commons import traverse_node, log_debug
from config import *
import numpy as np


def main_task():
    x1, x2, labels = get_prinicipal_features_and_labels()
    if ENABLE_PLOT:
        draw_scatter_plot(x1, x2, (0, 0), (0, 1), labels)
    x1 = np.array([x1]).transpose()
    x2 = np.array([x2]).transpose()
    log_debug("X1: ", x1.shape)
    log_debug("X2: ", x2.shape)
    log_debug("T: ", labels.shape)
    labels[labels == NEGATIVE_CLASS] = -1
    labels[labels == POSITIVE_CLASS] = 1
    x_nd = np.column_stack((x1, x2, labels))
    node = get_delta_and_tow(x_nd)
    stats_dict = {
        '{}'.format(NEGATIVE_CLASS_MAPPED): 0,
        '{}'.format(POSITIVE_CLASS_MAPPED): 0,
    }
    traverse_node(node, stats_dict)
    log_debug(stats_dict)


if __name__ == '__main__':
    main_task()
