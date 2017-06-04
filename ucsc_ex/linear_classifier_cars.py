import xlrd
import csv
import numpy as np
import numpy.linalg as lalg
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

WORKBOOK = xlrd.open_workbook('data/6/Car_Data.xlsx')
WORKSHEET = WORKBOOK.sheet_by_index(0)
TOTAL_COLS = WORKSHEET.ncols
TOTAL_ROWS = WORKSHEET.nrows
NUM_HDR_ROW = 2
NUM_ROWS = TOTAL_ROWS - NUM_HDR_ROW
NUM_TARGET_COLS = 1
NUM_FAKE_COL_W0 = 1
COL_IDX_TARGET_RECOMMENDATION = 6
NUM_COLS = TOTAL_COLS - NUM_TARGET_COLS

DEBUG_LOG = True
# np.set_printoptions(precision=4)
# np.set_printoptions(suppress=True)


def log_debug(*args):
    if DEBUG_LOG:
        print(*args)


def log(*args):
    print(*args)


def get_feature_vector(sheet, total_rows, num_rows, num_cols, num_hdr_row):
    log_debug("\nReading Features")
    log_debug("\tNum Feature Rows: {}".format(num_rows))
    log_debug("\tNum Feature Cols: {}".format(num_cols))
    price = sheet.col_values(0, NUM_HDR_ROW)
    maintenance = sheet.col_values(1, NUM_HDR_ROW)
    doors = sheet.col_values(2, NUM_HDR_ROW)
    persons = sheet.col_values(3, NUM_HDR_ROW)
    trunk = sheet.col_values(4, NUM_HDR_ROW)
    safety = sheet.col_values(5, NUM_HDR_ROW)
    return price, maintenance, doors, persons, trunk, safety


def read_training_feature_vector():
    return get_feature_vector(WORKSHEET, TOTAL_ROWS,
                              NUM_ROWS, NUM_COLS, NUM_HDR_ROW)


def read_target_recommendation_vector():
    recommendation = WORKSHEET.col_values(
        COL_IDX_TARGET_RECOMMENDATION, NUM_HDR_ROW)
    log_debug("\nT len:", len(recommendation))
    log_debug("\tT [0]: ", recommendation[0])
    log_debug("\tT [N]: ", recommendation[len(recommendation) - 1])
    return recommendation


def convert_nominal_to_kesler_target(feature_vec_x):
    keslerized_feature_vec = list()
    feature_vec = [str(x) for x in feature_vec_x]
    unique_value_set = list(sorted(set(feature_vec)))
    log_debug("T Set: ", unique_value_set)
    for idx in range(0, len(feature_vec)):
        row_kesler = np.full(shape=(len(unique_value_set)), fill_value=-1)
        feature_value_kesler_idx = unique_value_set.index(feature_vec[idx])
        row_kesler[feature_value_kesler_idx] = 1
        keslerized_feature_vec.append(row_kesler)
    # log_debug("\n", keslerized_feature_vec)
    return np.asarray(keslerized_feature_vec)


def convert_nominal_to_kesler_b(feature_vec_x):
    keslerized_feature_vec = list()
    feature_vec = [str(x) for x in feature_vec_x]
    unique_value_set = list(sorted(set(feature_vec)))
    log_debug("X Set: ", unique_value_set)
    for idx in range(0, len(feature_vec)):
        row_kesler = np.zeros(shape=(len(unique_value_set)), dtype=int)
        feature_value_kesler_idx = unique_value_set.index(str(feature_vec[idx]))
        row_kesler[feature_value_kesler_idx] = 1
        keslerized_feature_vec.append(row_kesler)
    # log_debug("\n", keslerized_feature_vec)
    return keslerized_feature_vec


"""
def convert_nominal_to_kesler(feature_vec):
    keslerized_feature_vec = list()
    unique_values = np.unique(feature_vec)
    unique_value_map = {}
    for unique_val_idx in range(0, len(unique_values)):
        unique_value_map[str(unique_values[unique_val_idx])] = unique_val_idx
    for idx in range(0, len(feature_vec)):
        row_kesler = np.zeros(shape=(len(unique_values)), dtype=int)
        feature_value_kesler_idx = unique_value_map[str(feature_vec[idx])]
        row_kesler[feature_value_kesler_idx] = 1
        keslerized_feature_vec.append(row_kesler)
    # log_debug("\n", keslerized_feature_vec)
    return keslerized_feature_vec
"""


def get_augmented_keslerized_features(price, maintenance, doors, persons, trunk, safety):
    price_kesler = convert_nominal_to_kesler_b(price)
    maint_kesler = convert_nominal_to_kesler_b(maintenance)
    doors_kesler = convert_nominal_to_kesler_b(doors)
    persons_kesler = convert_nominal_to_kesler_b(persons)
    trunk_kesler = convert_nominal_to_kesler_b(trunk)
    safety_kesler = convert_nominal_to_kesler_b(safety)
    return np.hstack((np.full(shape=(len(price), 1),
                              fill_value=1), price_kesler, maint_kesler,
                      doors_kesler, persons_kesler, trunk_kesler,
                      safety_kesler))


def get_keslerized_target(all_targets_raw):
    return convert_nominal_to_kesler_target(all_targets_raw)


def compute_weight_vector(xa, target):
    xa_pseudo_inv = lalg.pinv(xa)
    w = np.dot(xa_pseudo_inv, target)
    log_debug("\nW shape:", w.shape)
    log_debug(w)
    return w


def get_targets_keslerized(xa_vector, training_w):
    test_targets = np.zeros(shape=(len(xa_vector), training_w.shape[1]),
                            dtype=np.int)
    for xa_i_idx in range(0, len(xa_vector)):
        w_raw = np.dot(xa_vector[xa_i_idx], training_w)
        max_w_raw_idx = np.argmax(w_raw)
        xa_i_kesler = np.zeros(training_w.shape[1], dtype=np.int)
        xa_i_kesler.fill(-1)
        xa_i_kesler[max_w_raw_idx] = 1
        test_targets[xa_i_idx] = xa_i_kesler
    log_debug("\nTargets predicted kesler: ", test_targets.shape)
    log_debug("\tT/p [0]: ", test_targets[0])
    log_debug("\tT/p [N]: ", test_targets[len(test_targets) - 1])
    return test_targets


def convert_target_kesler_to_nominals(target_raw, target_kesler):
    target_raw_str = [str(x) for x in target_raw]
    unique_target_set = list(sorted(set(target_raw_str)))
    assert isinstance(target_kesler, np.ndarray)
    targets_nominal = [unique_target_set[np.argmax(t)] for t in target_kesler]
    targets = np.asarray(targets_nominal)
    return np.asarray(targets)


def convert_targets_to_binary(targets_raw):
    return ["unacc" if val == "unacc" else "acc" for val in targets_raw]


def get_keslerized_target_binary(targets_binary):
    return [-1 if val == "unacc" else 1 for val in targets_binary]


def get_targets_2d(xa_vector, training_w2d):
    test_targets = [(1 if (np.dot(xa_i, training_w2d) > 0) else -1) for xa_i in xa_vector]
    test_targets_2d = np.asarray(test_targets)
    log_debug("\nTargets2d Predicted: ", test_targets_2d.shape)
    return test_targets_2d


def main_task():
    price, maintenance, doors, persons, trunk, safety = \
        read_training_feature_vector()
    features_keslerized = get_augmented_keslerized_features(price,
                                                            maintenance, doors, persons, trunk, safety)
    log_debug("Features:")
    log_debug("\t X shape: ", features_keslerized.shape)
    n = len(features_keslerized)
    log_debug(features_keslerized[0])
    log_debug(features_keslerized[n - 1])

    targets_raw = read_target_recommendation_vector()
    targets_keslerized = get_keslerized_target(targets_raw)
    log_debug("Target: ")
    log_debug("\t T [0]: ", targets_keslerized[0])
    log_debug("\t T [N]: ", targets_keslerized[len(targets_keslerized) - 1])

    weights = compute_weight_vector(features_keslerized, targets_keslerized)
    targets_predicted_keslerized = get_targets_keslerized(
        features_keslerized, weights)
    targets_predicted = convert_target_kesler_to_nominals(
        targets_raw, targets_predicted_keslerized)
    cm = confusion_matrix(targets_raw, targets_predicted)
    log_debug("\nConfusion Matrix: ")
    log_debug("\t", cm)

    targets_binary = convert_targets_to_binary(targets_raw)
    targets_binary_keslerized = get_keslerized_target_binary(targets_binary)
    log_debug("\nTarget binary: ")
    log_debug("\t T [0]: ", targets_binary_keslerized[0])
    log_debug("\t T [N]: ", targets_binary_keslerized[len(
        targets_binary_keslerized) - 1])

    weights_binary = compute_weight_vector(features_keslerized,
                                           targets_binary_keslerized)
    targets_binary_predicted_keslerized = get_targets_2d(
        features_keslerized, weights_binary)
    log_debug("1: ", len(targets_binary_predicted_keslerized[
                  targets_binary_predicted_keslerized == 1]))
    targets_binary_predicted = convert_target_kesler_to_nominals(
        targets_binary, targets_binary_predicted_keslerized)

    cm = confusion_matrix(targets_binary_keslerized, targets_binary_predicted_keslerized)
    log_debug("\nConfusion Matrix: ")
    log_debug("\t", cm)


main_task()
