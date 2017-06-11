import xlrd
import numpy as np
import numpy.linalg as lalg

WORKBOOK = xlrd.open_workbook('data/5/Assignment_5_Data_and_Template.xlsx')
WORKSHEET = WORKBOOK.sheet_by_index(0)
INDEX_COL_FEET = 0
INDEX_COL_INCH = 1
INDEX_COL_GENDER = 2
DEBUG_LOG = True


def log_debug(*args):
    if DEBUG_LOG:
        print(*args)


def log(*args):
    print(*args)


def get_height_gender_array():
    feet_list = [col_val for col_val in WORKSHEET.col_values(INDEX_COL_FEET, 1)]
    inch_list = [col_val for col_val in WORKSHEET.col_values(INDEX_COL_INCH, 1)]
    gender_list = [-1 if col_val == "Female" else 1 for col_val in
                   WORKSHEET.col_values(INDEX_COL_GENDER, 1)]
    height_list = [(feet * 12 + inch) for feet, inch in zip(feet_list, inch_list)]
    feature_matrix_raw = np.column_stack(zip(height_list, gender_list))
    feature_matrix = feature_matrix_raw.transpose()
    return feature_matrix[feature_matrix[:, 0].argsort(kind='mergesort')]


def get_impurity_improvement_and_optimal_split(x_nd, x_all, t_all):
    x_count = len(x_all)
    x_negative = [x_nd[x, 0] for x in range(0, x_count) if x_nd[x, 1] == -1]
    t_negative = len(x_negative)
    x_positive = [x_nd[x, 0] for x in range(0, x_count) if x_nd[x, 1] == 1]
    t_positive = len(x_positive)
    log_debug(x_count, t_negative, t_positive)

    a_negative = 0
    a_positive = 0
    impurity_initial = impurity_optimal = (t_negative * t_positive) / (x_count * x_count)
    tow = x_all[0]

    for idx in range(1, x_count):
        if t_all[idx - 1] == -1:
            a_negative += 1
        else:
            a_positive += 1

        impurity_part2_1 = ((a_negative * a_positive) / (a_negative + a_positive))
        impurity_part2_2 = ((t_negative - a_negative) * (t_positive - a_positive)) / (
            t_negative + t_positive - a_negative - a_positive)
        impurity_tmp = (1 / x_count) * (impurity_part2_1 + impurity_part2_2)
        if impurity_tmp < impurity_optimal:
            impurity_optimal = impurity_tmp
            tow = x_all[idx]
            tow_idx = idx

    log_debug("Io: ", impurity_initial)
    log_debug("Iopt: ", impurity_optimal)
    log_debug("I delta: ", impurity_initial - impurity_optimal)
    log_debug("Tow: ", tow, ", (i={})".format(tow_idx), x_nd[tow_idx])


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


def main_task():
    x_nd = get_height_gender_array()
    x_all = [x_nd[x, 0] for x in range(0, len(x_nd))]
    t_all = [x_nd[x, 1] for x in range(0, len(x_nd))]
    get_impurity_improvement_and_optimal_split(x_nd, x_all, t_all)
    get_linear_classifier_weights(x_all, t_all)

main_task()


