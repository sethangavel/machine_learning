import xlrd
import numpy as np
import numpy.linalg as lalg
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


def log_debug(*args):
    if DEBUG_LOG:
        print(*args)


def log(*args):
    print(*args)


def read_feature_vector_impl(sheet, num_rows, num_cols, num_hdr_row):
    log_debug("\nReading Features")
    log_debug("\tNum Feature Rows: {}".format(num_rows))
    log_debug("\tNum Feature Cols: {}".format(num_cols))
    price = sheet.col_values(0, num_hdr_row)
    maintenance = sheet.col_values(1, num_hdr_row)
    doors = sheet.col_values(2, num_hdr_row)
    persons = sheet.col_values(3, num_hdr_row)
    trunk = sheet.col_values(4, num_hdr_row)
    safety = sheet.col_values(5, num_hdr_row)
    return price, maintenance, doors, persons, trunk, safety


def read_training_feature_vector():
    return read_feature_vector_impl(WORKSHEET, NUM_ROWS, NUM_COLS, NUM_HDR_ROW)


def read_target_recommendation_vector():
    recommendation = WORKSHEET.col_values(
        COL_IDX_TARGET_RECOMMENDATION, NUM_HDR_ROW)
    log_debug("\nT len:", len(recommendation))
    log_debug("\tT [0]: ", recommendation[0])
    log_debug("\tT [N]: ", recommendation[len(recommendation) - 1])
    return recommendation


def convert_target_nominal_to_kesler(feature_vec_x):
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


def convert_feature_nominal_to_kesler(feature_vec_x):
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


def get_augmented_features_keslerized(price, maintenance, doors, persons,
                                      trunk, safety):
    price_kesler = convert_feature_nominal_to_kesler(price)
    maint_kesler = convert_feature_nominal_to_kesler(maintenance)
    doors_kesler = convert_feature_nominal_to_kesler(doors)
    persons_kesler = convert_feature_nominal_to_kesler(persons)
    trunk_kesler = convert_feature_nominal_to_kesler(trunk)
    safety_kesler = convert_feature_nominal_to_kesler(safety)
    return np.hstack((np.full(shape=(len(price), 1),
                              fill_value=1), price_kesler, maint_kesler,
                      doors_kesler, persons_kesler, trunk_kesler,
                      safety_kesler))


def convert_target_multi_class_to_kesler(all_targets_raw):
    return convert_target_nominal_to_kesler(all_targets_raw)


def compute_weight_vector(xa, target):
    xa_pseudo_inv = lalg.pinv(xa)
    w = np.dot(xa_pseudo_inv, target)
    log_debug("\nW shape:", w.shape)
    log_debug(w)
    return w


def get_targets_multi_class_keslerized(xa_vector, training_w):
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


def convert_target_multi_class_kesler_to_nominals(target_raw, target_kesler):
    target_raw_str = [str(x) for x in target_raw]
    unique_target_set = list(sorted(set(target_raw_str)))
    assert isinstance(target_kesler, np.ndarray)
    targets_nominal = [unique_target_set[np.argmax(t)] for t in target_kesler]
    targets = np.asarray(targets_nominal)
    return np.asarray(targets)


def convert_targets_multi_class_to_binary(targets_raw):
    targets = list()
    for val in targets_raw:
        if val == "unacc":
            targets.append("unacc")
        else:
            targets.append("acc")
    return targets


def convert_target_binary_to_kesler(targets_binary):
    return [-1 if val == "unacc" else 1 for val in targets_binary]


def convert_target_binary_kesler_to_nominals(targets_binary_predicted_keslerized):
    targets = list()
    for val in targets_binary_predicted_keslerized:
        if val == -1:
            targets.append("unacc")
        else:
            targets.append("acc")
    return targets


def get_targets_binary(xa_vector, training_w2d):
    test_targets = [(1 if (np.dot(xa_i, training_w2d) > 0) else -1) for xa_i in xa_vector]
    test_targets_2d = np.asarray(test_targets)
    log_debug("\nTargets2d Predicted: ", test_targets_2d.shape)
    return test_targets_2d


def main_task():
    price, maintenance, doors, persons, trunk, safety = \
        read_training_feature_vector()
    features_keslerized = get_augmented_features_keslerized(
        price, maintenance, doors, persons, trunk, safety)
    log_debug("Features:")
    log_debug("\t X shape: ", features_keslerized.shape)
    n = len(features_keslerized)
    log_debug(features_keslerized[0])
    log_debug(features_keslerized[n - 1])

    targets_multi_class_raw = read_target_recommendation_vector()
    targets_multi_class_keslerized = convert_target_multi_class_to_kesler(
        targets_multi_class_raw)
    log_debug("Target: ")
    log_debug("\t T [0]: ", targets_multi_class_keslerized[0])
    log_debug("\t T [N]: ", targets_multi_class_keslerized[len(
                  targets_multi_class_keslerized) - 1])

    weights_multi_class = compute_weight_vector(features_keslerized,
                                                targets_multi_class_keslerized)
    targets_multi_class_predicted_keslerized = get_targets_multi_class_keslerized(
        features_keslerized, weights_multi_class)
    targets_multi_class_predicted =\
        convert_target_multi_class_kesler_to_nominals(targets_multi_class_raw,
                                                      targets_multi_class_predicted_keslerized)
    cm = confusion_matrix(targets_multi_class_raw, targets_multi_class_predicted)
    log_debug("\nConfusion Matrix: ")
    log_debug("\t", cm)

    targets_binary = convert_targets_multi_class_to_binary(
        targets_multi_class_raw)
    targets_binary_keslerized = convert_target_binary_to_kesler(
        targets_binary)
    log_debug("\nTarget binary: ")
    log_debug("\t T [0]: ", targets_binary_keslerized[0])
    log_debug("\t T [N]: ", targets_binary_keslerized[len(
        targets_binary_keslerized) - 1])

    weights_binary = compute_weight_vector(features_keslerized,
                                           targets_binary_keslerized)
    targets_binary_predicted_keslerized = get_targets_binary(
        features_keslerized, weights_binary)
    log_debug("Predicted Accetped Kesler: ", len(targets_binary_predicted_keslerized[
                  targets_binary_predicted_keslerized == 1]))
    log_debug("Predicted UnAccetped Kesler: ", len(targets_binary_predicted_keslerized[
                                                     targets_binary_predicted_keslerized == -1]))
    targets_binary_predicted = convert_target_binary_kesler_to_nominals(
        targets_binary_predicted_keslerized)

    cm_sk_1 = confusion_matrix(targets_binary, targets_binary_predicted, labels=("unacc", "acc"))
    cm_sk_2 = confusion_matrix(targets_binary_keslerized, targets_binary_predicted_keslerized)
    log_debug("\nConfusion Matrix: ")
    log_debug("\tcm by nominals:", cm_sk_1)
    log_debug("\tcm by kesler:", cm_sk_2)


main_task()
