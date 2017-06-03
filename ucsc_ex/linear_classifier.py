
import xlrd
import csv
import numpy as np
import numpy.linalg as lalg
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

"""
Training DataSet
"""
WORKBOOK = xlrd.open_workbook('data/4/Assignment_4_Data_and_Template.xlsx')
TRAINING_SHEET = WORKBOOK.sheet_by_index(0)
TRAINING_TOTAL_COLS = TRAINING_SHEET.ncols
TRAINING_TOTAL_ROWS = TRAINING_SHEET.nrows
TRAINING_NUM_HDR_ROW = 1
TRAINING_NUM_ROWS = TRAINING_TOTAL_ROWS - TRAINING_NUM_HDR_ROW
TRAINING_NUM_TARGET_COLS = 2
TRAINING_NUM_FAKE_COL_W0 = 1
TRAINING_COL_IDX_TARGET_FAILURE = 15
TRAINING_COL_IDX_TARGET_TYPE = 16
TRAINING_NUM_COLS = TRAINING_TOTAL_COLS - TRAINING_NUM_TARGET_COLS\
                    + TRAINING_NUM_FAKE_COL_W0

"""
Test DataSet
"""
TEST_SHEET = WORKBOOK.sheet_by_index(2)
TEST_TOTAL_COLS = TEST_SHEET.ncols
TEST_TOTAL_ROWS = TEST_SHEET.nrows
TEST_NUM_HDR_ROW = 4
TEST_NUM_ROWS = TEST_TOTAL_ROWS - TEST_NUM_HDR_ROW
TEST_NUM_TARGET_COLS = 2
TEST_NUM_FAKE_COL_W0 = 1
TEST_COL_IDX_TARGET_FAILURE = 15
TEST_COL_IDX_TARGET_TYPE = 16
TEST_NUM_COLS = TEST_TOTAL_COLS - TEST_NUM_TARGET_COLS + TEST_NUM_FAKE_COL_W0


DEBUG_LOG = True
OUT_CSV_FILE = "out/linear_classifier.csv"
OUT_CSV_FD = None
OUT_STREAM = None
OUT_SKIP = False

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


def log_debug(*args):
    if DEBUG_LOG:
        print(*args)


def log(*args):
    print(*args)


def get_augmented_feature_vector(sheet, total_rows, num_rows, num_cols,
                                 num_hdr_row, fake_cols, context):
    log_debug("\nReading ", context)
    log_debug("\t[{}] Num Feature Rows: {}".format(context, num_rows))
    log_debug("\t[{}] Num Feature Cols: {}".format(context, num_cols))
    all_features = np.zeros(shape=(num_rows, num_cols))
    for row_idx in range(num_hdr_row, total_rows):
        row = sheet.row_values(row_idx, 0, num_cols - fake_cols)
        row.insert(0, 1)
        all_features[row_idx - num_hdr_row] = row
    log_debug("\t[{}] Xa shape: {}".format(context, all_features.shape))
    log_debug("\t[{}] Xa [0]: {}".format(context, all_features[0]))
    log_debug("\t[{}] Xa [N]: {}".format(context,
                                         all_features[len(all_features) - 1]))
    return all_features


def read_training_augmented_feature_vector():
    return get_augmented_feature_vector(TRAINING_SHEET, TRAINING_TOTAL_ROWS,
                                        TRAINING_NUM_ROWS, TRAINING_NUM_COLS,
                                        TRAINING_NUM_HDR_ROW,
                                        TRAINING_NUM_FAKE_COL_W0, "Training")


def read_test_augmented_feature_vector():
    return get_augmented_feature_vector(TEST_SHEET, TEST_TOTAL_ROWS,
                                        TEST_NUM_ROWS, TEST_NUM_COLS,
                                        TEST_NUM_HDR_ROW, TEST_NUM_FAKE_COL_W0,
                                        "Test")


def read_target_failure_scalar():
    target_failure_list = TRAINING_SHEET.col_values(
        TRAINING_COL_IDX_TARGET_FAILURE, TRAINING_NUM_HDR_ROW)
    target_failure = np.asarray(target_failure_list, dtype=np.int)
    log_debug("\nReading Target Failure")
    log_debug("\tT2 shape: ", target_failure.shape)
    log_debug("\tT2 [0]: ", target_failure[0])
    log_debug("\tT2 [N]: ", target_failure[len(target_failure) - 1])
    return target_failure


def read_target_type_scalar():
    target_type_list = TRAINING_SHEET.col_values(TRAINING_COL_IDX_TARGET_TYPE,
                                                 TRAINING_NUM_HDR_ROW)
    targets = np.asarray(target_type_list, dtype=np.int)
    log_debug("\nReading Target Type:")
    log_debug("\tT6 shape: ", targets.shape)
    log_debug("\tT6 [0]: ", targets[0])
    log_debug("\tT6 [N]: ", targets[len(targets) - 1])
    return targets


def convert_scalar_to_6d_kesler(t6_targets):
    target_type = np.zeros(shape=(len(t6_targets), 6), dtype=np.int)
    for t_idx in range(0, len(t6_targets)):
        t = int(t6_targets[t_idx])
        t_kesler = np.zeros(6, dtype=np.int)
        t_kesler.fill(-1)
        t_kesler[t] = 1
        target_type[t_idx] = t_kesler
    log_debug("\nConvert target Type")
    log_debug("\tT6 shape: ", target_type.shape)
    log_debug("\tT6 [0]: ", target_type[0])
    log_debug("\tT6 [N]: ", target_type[len(target_type) - 1])
    return target_type


def compute_weight_vector(xa, target):
    xa_pseudo_inv = lalg.pinv(xa)
    w = np.dot(xa_pseudo_inv, target)
    write_file_ndarray("W", w)
    return w


def get_targets_2d(xa_vector, training_w2d, do_print):
    test_targets = [(1 if (np.dot(xa_i, training_w2d) > 0) else -1) for xa_i in xa_vector]
    test_targets_2d = np.asarray(test_targets)
    log_debug("\nTargets2d Predicted: ", test_targets_2d.shape)
    if do_print:
        write_file_ndarray("Targets Failure:", np.asarray(
            test_targets).reshape(50, 1))
        # for idx in range(0, len(xa_vector)):
        #    log_debug("Xa_i: ", idx, " : ", test_targets[idx])
    return test_targets_2d


def get_targets_6d_kesler(xa_vector, training_w):
    test_targets = np.zeros(shape=(len(xa_vector), training_w.shape[1]),
                            dtype=np.int)
    for xa_i_idx in range(0, len(xa_vector)):
        w_raw = np.dot(xa_vector[xa_i_idx], training_w)
        max_w_raw_idx = np.argmax(w_raw)
        xa_i_kesler = np.zeros(training_w.shape[1], dtype=np.int)
        xa_i_kesler.fill(-1)
        xa_i_kesler[max_w_raw_idx] = 1
        test_targets[xa_i_idx] = xa_i_kesler
    log_debug("\nTargets6d predicted kesler: ", test_targets.shape)
    # log_debug("\tT6/p [0]: ", test_targets[0])
    # log_debug("\tT6/p [N]: ", test_targets[len(test_targets) - 1])
    return test_targets


def convert_6d_kesler_to_scalar(t6_training_kesler, do_print):
    assert isinstance(t6_training_kesler, np.ndarray)
    targets_raw = [np.argmax(t) for t in t6_training_kesler]
    targets = np.asarray(targets_raw)
    log_debug("Convert targets 6D: ", targets.shape)
    if do_print:
        write_file_ndarray("Targets Type: ", np.asarray(targets).reshape(50, 1))
    return np.asarray(targets)


def get_accuracy_2d(actual_truth, predicted_truth):
    positive = 1
    negative = -1
    true_positive = true_negative = false_positive = false_negative = 0
    for actual_i, predicted_i in zip(actual_truth, predicted_truth):
        if actual_i == positive:
            if predicted_i == actual_i:
                true_positive += 1
            else:
                false_negative += 1
        elif actual_i == negative:
            if predicted_i == actual_i:
                true_negative += 1
            else:
                false_positive += 1
    total_samples = len(actual_truth)
    accuracy = (true_positive + true_negative) / (1.0 * total_samples)
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    positive_predictive_value = true_positive / (true_positive + false_positive)
    log_debug("\n2D")
    log_debug("\tAccuracy: ", accuracy)
    log_debug("\tAccuracy(SK): ", accuracy_score(actual_truth, predicted_truth))
    log_debug("\tSensitivity: ", sensitivity)
    log_debug("\tSpecificity: ", specificity)
    log_debug("\tPPV: ", positive_predictive_value)
    # return [[true_positive, false_negative], [false_positive, true_negative]]
    return [[true_negative, false_positive], [false_negative, true_positive]]


def get_accuracy_6d(actual_truth, predicted_truth):
    accuracy = accuracy_score(actual_truth, predicted_truth)
    log_debug("\n6D")
    log_debug("\tAccuracy(SK): ", accuracy)


def get_ppv_6d(conf_mat):
    ppvs = {}
    for row_i in range(0, len(conf_mat)):
        val = conf_mat[row_i][row_i] / np.sum(conf_mat[:, row_i])
        ppvs[val] = row_i
    for sorted_val in sorted(ppvs.keys()):
        log_debug(sorted_val, " : ", ppvs[sorted_val])
    return ppvs


def open_out_file():
    global OUT_CSV_FD
    global OUT_STREAM
    OUT_CSV_FD = open(OUT_CSV_FILE, 'w')
    OUT_STREAM = csv.writer(OUT_CSV_FD, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL,
                            lineterminator='\n')


def close_out_file():
    global OUT_CSV_FD
    OUT_CSV_FD.close


def write_file_array(l_msg, *args):
    if OUT_SKIP:
        return
    global OUT_STREAM
    row_title = [l_msg]
    data = [*args]
    log_debug(row_title, ": ", data)
    if len(l_msg) > 0:
        OUT_STREAM.writerow(row_title)
    for row in data:
        if isinstance(row, int) or isinstance(row, np.float64):
            row = [row]
        OUT_STREAM.writerow(row)


def write_file_ndarray(l_msg, ndarray_arg):
    assert isinstance(ndarray_arg, np.ndarray)
    OUT_STREAM.writerow([l_msg])
    for row_array in ndarray_arg:
        write_file_array("", row_array)


open_out_file()
Xa_training = read_training_augmented_feature_vector()
T2_training = read_target_failure_scalar()
W2D_training = compute_weight_vector(Xa_training, T2_training)
T2_training_predicted = get_targets_2d(Xa_training, W2D_training, False)
conf_mat_2d = get_accuracy_2d(T2_training, T2_training_predicted)
conf_mat_2d_sk = confusion_matrix(T2_training, T2_training_predicted)
log_debug("Confusion Matrix 2D: ", conf_mat_2d)
log_debug("Confusion Matrix 2D SK: ", conf_mat_2d_sk)

T6_training = read_target_type_scalar()
T6_training_kesler = convert_scalar_to_6d_kesler(T6_training)
assert((T6_training == convert_6d_kesler_to_scalar(T6_training_kesler,
                                                   False)).all())

W6D_training = compute_weight_vector(Xa_training, T6_training_kesler)
T6_training_predicted_kesler = get_targets_6d_kesler(Xa_training, W6D_training)
T6_training_predicted = convert_6d_kesler_to_scalar(
    T6_training_predicted_kesler, False)
get_accuracy_6d(T6_training_kesler, T6_training_predicted_kesler)
get_accuracy_6d(T6_training, T6_training_predicted)

Xa_test = read_test_augmented_feature_vector()
T2_test = get_targets_2d(Xa_test, W2D_training, True)
T6_test_kesler = get_targets_6d_kesler(Xa_test, W6D_training)
T6_test = convert_6d_kesler_to_scalar(T6_test_kesler, True)
conf_mat_6d_sk = confusion_matrix(T6_training, T6_training_predicted)
write_file_ndarray("Confusion Matrix 6D: ", conf_mat_6d_sk)
get_ppv_6d(conf_mat_6d_sk)




