#!/usr/bin/python

from config import *
from commons import log_debug

import csv
import struct
import matplotlib as plt
from array import array as pyarray
from pylab import *
from numpy import *
import numpy.linalg as la
from sklearn.decomposition import PCA as sk_pca
import os

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def load_mnist(dataset="training", digits=range(10), path='data/3/'):
    """
    Adapted from: http://cvxopt.org/applications/svm/index.html?highlight=mnist
    """
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx')
        fname_lbl = os.path.join(path, 't10k-labels.idx')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ind[i]*rows*cols: (ind[i]+1)*rows*cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def draw_image(byte_array, img_title):
    img_byte_array_len = len(np.ravel(byte_array))
    dim = int(np.sqrt(img_byte_array_len))
    if not ENABLE_IMAGE_SHOW:
        return
    plt.imshow(byte_array.reshape(dim, dim), interpolation='None', cmap=cm.gray)
    plt.title(img_title)
    show()


def draw_image_subplot(byte_array, img_title, sub_plt):
    if not ENABLE_IMAGE_SHOW:
        return
    sub_plt.imshow(byte_array.reshape(28, 28), interpolation='None', cmap=cm.gray)
    sub_plt.set_title(img_title)
    sub_plt.axis('off')


def draw_scatter_plot(cloud_a, cloud_b, xp, xn, all_labels):
    if not ENABLE_PLOT:
        return
    log_debug("\nPC Scatter Plot: {}, {}!".format(xp, xn))
    fig = plt.figure()
    cols = np.zeros((alen(all_labels), 4))
    for idx, ll in enumerate(all_labels):
        if ll == POSITIVE_CLASS:
            cols[idx] = [1, 0, 0, SCATTER_PLOT_ALPHA]
        if ll == NEGATIVE_CLASS:
            cols[idx] = [0, 0.2, 1, SCATTER_PLOT_ALPHA]
    random_order = np.arange(np.alen(all_labels))
    ax = fig.add_subplot(111, facecolor='white')
    ax.scatter(cloud_b, cloud_a, s=8, linewidths=0,
               facecolors=cols[random_order, :], marker='o')
    ax.plot(xp[1], xp[0], marker='x', color='black', label='XP[{}]'.format(POSITIVE_CLASS))
    ax.plot(xn[1], xn[0], marker='*', color='black', label='XN[{}]'.format(NEGATIVE_CLASS))
    ax.legend(loc='lower left', numpoints=1, ncol=3, fontsize=10, bbox_to_anchor=(0, 0))
    ax.set_aspect('equal')
    plt.rcParams['axes.facecolor'] = 'b'
    plt.gca().invert_yaxis()
    plt.title('Principal Components PC1 and PC2 scatter plot\nManoj Govindassamy')
    plt.show()


# converting from NX28X28 array into NX784 array
def get_x_feature_vectors(images_3d_array):
    flat_images = list()
    for i in images_3d_array:
        flat_images.append(i.ravel())
    x_feature_vectors = np.asarray(flat_images)
    log_debug("X shape: ", x_feature_vectors.shape)
    log_debug("X min/max: ", np.amin(x_feature_vectors), np.amax(x_feature_vectors))
    return x_feature_vectors


def get_mu_mean_vectors(x_feature_vector):
    mu = np.mean(x_feature_vector, axis=0, dtype=np.float64)
    log_debug("MU shape:", mu.shape)
    return mu


def get_z_variance_vector(x_feature_vector, mu_mean_vector):
    z = np.subtract(x_feature_vector, mu_mean_vector)
    log_debug("Z shape: ", z.shape)
    return z


def get_c_covariance_vector(z_variance_vector):
    c = np.cov(z_variance_vector, rowvar=False, ddof=1)
    log_debug("C shape: ", c.shape)
    # draw_image(c, "Cov")
    return c


def verify_c_covariance_vector(c_covariance_vector):
    c_row = c_covariance_vector.shape[0]
    c_col = c_covariance_vector.shape[1]
    assert c_row == c_col
    for idx_r in range(c_row):
        for idx_c in range(c_col):
            if idx_r == idx_c:
                assert c_covariance_vector[idx_r][idx_c] >= 0
            assert c_covariance_vector[idx_r][idx_c] == c_covariance_vector[idx_c][idx_r]
    log_debug("C Verification: Good!")
    return


def get_eigen_value_n_vector(c_covariance_vector):
    [eig_val, eig_vec] = la.eigh(c_covariance_vector)
    log_debug("EigVal len: ", len(eig_val))
    log_debug("EigVec shape: ", eig_vec.shape)
    return [eig_val, eig_vec]


def verify_v_eigen_value_n_vector(eig_val, v_vec):
    for row in v_vec:
        if round(np.linalg.norm(row)) != 1.0:
            log_debug(np.linalg.norm(row))
    assert round(np.dot(v_vec[10], v_vec[100]), 5) == 0


def is_eigen_values_row_aligned(c_covariance_matrix, eig_val, eig_vec, row, col):
    # log_debug(eig_val)
    # log_debug(np.dot(c_covariance_matrix, row) / (eig_val[0] * row))
    # log_debug(np.dot(c_covariance_matrix, col) / (eig_val[0] * col))
    sum_x = round(sum(np.dot(c_covariance_matrix, eig_vec[0]) - np.dot(eig_val[0], eig_vec[0])), 8)
    log_debug("EigVec Verifi: ", sum_x)
    if sum_x == 0.0:
        return True
    return False


def verify_pca_vector(p_pca_vector):
    assert round(max(abs(np.mean(p_pca_vector, axis=0))), 5) == 0
    log_debug("P shape: ", p_pca_vector.shape)


def verify_images_and_plots(x_matrix, mu_vec, z_vec, c_vec, v_eig_vector, p_pca_vector,
                            all_labels, image_index):
    log_debug("Verify Images!")
    if not ENABLE_IMAGE_SHOW:
        return

    fig = plt.figure()
    draw_image_subplot(x_matrix[image_index], "X img {}".format(image_index), fig.add_subplot(241))

    draw_image_subplot(v_eig_vector[0], "EigVec 0", fig.add_subplot(242))
    draw_image_subplot(v_eig_vector[1], "EigVec 1", fig.add_subplot(243))

    x_rec1 = (np.dot(p_pca_vector[:, 0:1], v_eig_vector[0:1, :])) + mu_vec
    draw_image_subplot(x_rec1[image_index], "Rec with Pc1", fig.add_subplot(244))

    x_rec2 = (np.dot(p_pca_vector[:, 0:10], v_eig_vector[0:10, :])) + mu_vec
    draw_image_subplot(x_rec2[image_index], "Rec with Pc10", fig.add_subplot(245))

    x_rec2 = (np.dot(p_pca_vector[:, 0:100], v_eig_vector[0:100, :])) + mu_vec
    draw_image_subplot(x_rec2[image_index], "Rec with Pc100", fig.add_subplot(246))

    x_rec = (np.dot(p_pca_vector, v_eig_vector)) + mu_vec
    draw_image_subplot(x_rec[image_index], "Rec with all PC", fig.add_subplot(247))

    fig.tight_layout(pad=0)
    show()


def get_pc_with_labels(pc_vector, label_list):
    pc_idx_0 = pc_vector[:, 0]
    pc_idx_1 = pc_vector[:, 1]
    return [pc_idx_0, pc_idx_1, label_list]


def get_pca_xi(xi, mu, zz):
    z = get_z_variance_vector(xi, mu)
    c = get_c_covariance_vector(zz)
    verify_c_covariance_vector(c)

    [eig_val, v] = get_eigen_value_n_vector(c)
    verify_v_eigen_value_n_vector(eig_val, v)
    eig_val = np.flipud(eig_val)
    v = np.flipud(v.T)
    p_xi = np.dot(z, v.T)

    write_file_array("(74) xp: ", xi)
    write_file_array("(75) zp: ", z)
    write_file_array("(76) pp: ", p_xi[0:2])

    r_xi = np.dot(p_xi[0:2], v[0:2])
    xrec_xi = r_xi + mu
    write_file_array("(77) rp: ", r_xi)
    write_file_array("(78) xrecp: ", xrec_xi)

    return p_xi[0:2]


def get_pca():
    images, l_labels = load_mnist('training', digits=[NEGATIVE_CLASS, POSITIVE_CLASS])

    x = get_x_feature_vectors(images)
    mu = get_mu_mean_vectors(x)
    z = get_z_variance_vector(x, mu)
    c = get_c_covariance_vector(z)
    verify_c_covariance_vector(c)

    [eig_val, v] = get_eigen_value_n_vector(c)
    verify_v_eigen_value_n_vector(eig_val, v)
    eig_row = v[0, :]
    eig_col = v[:, 0]

    if is_eigen_values_row_aligned(c, eig_val, v, eig_row, eig_col) and not FORCE_EIGEN_FLIP:
        log_debug("EigVec is already ROW aligned")
    else:
        log_debug("EigVec is COL aligned")
        eig_val = np.flipud(eig_val)
        v = np.flipud(v.T)
        if is_eigen_values_row_aligned(c, eig_val, v, eig_row, eig_col):
            log_debug("EigVec is already ROW aligned")
        else:
            assert "EigVec failed to be ROW aligned!"

    p = np.dot(z, v.T)
    verify_pca_vector(p)
    return [x, mu, z, c, v, p, l_labels]


def get_cube_root(n):
    return n ** (1.0 / 3.0)


def get_optimal_bin_count(min_samples):
    return get_optimal_bin_count_custom(min_samples)


def get_optimal_bin_count_sturges(min_samples):
    return math.ceil(np.log2(min_samples) + 1)


def get_optimal_bin_count_rice(min_samples):
    return math.ceil(2 * get_cube_root(min_samples))


def get_optimal_bin_count_custom(min_samples):
    return 25


def get_bin(xi, xmin, xmax):
    return int(round((bin_count - 1) * ((xi - xmin) / (xmax - xmin))))


def get_histo_matrix_row_col(h, s):
    r = get_bin(h, min_pc_1, max_pc_1)
    c = get_bin(s, min_pc_2, max_pc_2)
    return [r, c]


def get_histo_matrix(pi1, pi2, bin_counts):
    histo = np.zeros((bin_counts, bin_counts))
    for h, s in zip(pi1, pi2):
        rc_tup = get_histo_matrix_row_col(h, s)
        histo[rc_tup[0]][rc_tup[1]] = histo[rc_tup[0]][rc_tup[1]] + 1
    return histo


def get_2d_pdf(xi, mu_x, cov_x, ni_samples):
    log_debug("\tN: ", ni_samples)
    xi_mu = np.array([np.subtract(xi, mu_x)])
    cov_inverse = inv(cov_x)
    log_debug("\tCov Inv: ", cov_inverse)
    xi_mu_transpose = np.transpose(xi_mu)
    log_debug("\tMu Trans: ", xi_mu_transpose)
    xi_scalar = (np.dot(xi_mu, cov_inverse).dot(xi_mu_transpose)) / 2
    part_2 = math.exp(-1 * xi_scalar)
    cov_determinant = det(cov_x)
    part_1 = ni_samples / (2 * math.pi * math.sqrt(cov_determinant))
    log_debug("\tC Det: ", cov_determinant)
    log_debug("\traised: ", xi_scalar)
    log_debug("\tbottom: ", part_1/ni_samples)
    log_debug("\tpdf: ", part_1 * part_2)
    return part_1 * part_2


def print_histo_samples():
    log_debug("\nSample Size:")
    write_file_array("(17) Min_pc1  Max_pc1: ", [min_pc_1, max_pc_1])
    write_file_array("(18) Min_pc2  Max_pc2: ", [min_pc_2, max_pc_2])
    write_file_array("(19) Optimal bin count: ", bin_count)


def get_prob_by_histo(h_p, h_n, xi, mu_vec, v_vec, msg):
    log_debug("\n\n", msg)
    log_debug("Xi.shape: ", xi.shape)
    draw_image(xi, msg)
    zi_vec = get_z_variance_vector(xi, mu_vec)
    log_debug("Zi.shape: ", zi_vec.shape)
    vi_2d_vec = v_vec[0:2, :]
    log_debug("Vi.shape: ", vi_2d_vec.shape)
    pi_vec = np.dot(zi_vec, vi_2d_vec.T)
    log_debug("\t", pi_vec)
    r = get_bin(pi_vec[0], min_pc_1, max_pc_1)
    c = get_bin(pi_vec[1], min_pc_2, max_pc_2)
    xi_count = h_p[r][c]
    xj_count = h_n[r][c]
    xi_p = xi_count / (xi_count + xj_count)
    xi_n = xj_count / (xi_count + xj_count)
    log_debug("\tR:", r, ", C:", c, "i_Histo: ", xi_count, "j_Histo: ", xj_count)
    if xi_count + xj_count == 0:
        return "Undecidable!"
    else:
        return [xi_p, xi_n]


def get_bayes_2d_pdf(mu_i_vec, cov_i_vec, pc_i_vec, pi_vec, msg_str):
    log_debug(msg_str)
    ni_samples = len(pc_i_vec)
    return get_2d_pdf(pi_vec, mu_i_vec, cov_i_vec, ni_samples)


def get_xi_prob_by_histo(pos_pc_1, pos_pc_2, neg_pc_1, neg_pc_2, binc, xp, xn):
    log("\n\n\n")
    log("HISTO")
    log("-----")
    log_debug("\tClass +ve Histo:")
    pos_class_histo = get_histo_matrix(pos_pc_1, pos_pc_2, binc)
    draw_image(pos_class_histo, "+ve Histo")
    log("\tClass +ve Histo: ", sum(pos_class_histo))
    write_file_ndarray("(20) Hp [{}]".format(POSITIVE_CLASS), pos_class_histo)

    # pos_class_histo_npformula = np.histogram2d(pos_pc_1, pos_pc_2, binc)[0]
    # draw_image(pos_class_histo_npformula, "+ve Histo")

    log_debug("\tClass -ve Histo:")
    neg_class_histo = get_histo_matrix(neg_pc_1, neg_pc_2, binc)
    draw_image(neg_class_histo, "-ve Histo")
    log("\tClass -ve Histo: ", sum(neg_class_histo))
    write_file_ndarray("(46) Hn [{}]".format(NEGATIVE_CLASS), neg_class_histo)

    # neg_class_histo_npformula = np.histogram2d(neg_pc_1, neg_pc_2, binc)[0]
    # draw_image(neg_class_histo_npformula, "-ve Histo")

    xp_prob_histo = get_prob_by_histo(pos_class_histo, neg_class_histo,
                                      xp, MU, V, "Histo XP{} +ve".format(XP_INDEX))
    xn_prob_histo = get_prob_by_histo(pos_class_histo, neg_class_histo,
                                      xn, MU, V, "Histo XN{} -ve".format(XN_INDEX))
    log("\tXP : ", XP_INDEX, ", truth: ", labels[XP_INDEX], ", Prob Histo {}: {}".format(
        POSITIVE_CLASS, xp_prob_histo[0]))
    write_file_array("(89) Result of classifying xp using histograms: ",
                     [labels[XP_INDEX], xp_prob_histo[0]])
    log("\tXn : ", XN_INDEX, ", truth: ", labels[XN_INDEX], ", Prob Histo {}: {}".format(
        NEGATIVE_CLASS, xn_prob_histo[1]))
    write_file_array("(93) Result of classifying xn using histograms: ",
                     [labels[XN_INDEX], xn_prob_histo[1]])
    return [pos_class_histo, neg_class_histo]


def get_xi_prob_by_bayes(pos_pcs, neg_pcs, xp_pc, xn_pc):
    log("\n\n\n")
    log("BAYES")
    log("-----")
    log("\tBayes Query: \n", xp_pc, "\n", xn_pc)
    mu_p = get_mu_mean_vectors(pos_pcs)
    mu_n = get_mu_mean_vectors(neg_pcs)
    log_debug("\tClass +ve Bayesian Mu: ", mu_p, ", Class -ve Bayesian Mu: ", mu_n)
    write_file_array("(9) mup [{}]".format(POSITIVE_CLASS), mu_p)
    write_file_array("(10) mun [{}]".format(NEGATIVE_CLASS), mu_n)

    c_p = get_c_covariance_vector(pos_pcs)
    verify_c_covariance_vector(c_p)
    c_n = get_c_covariance_vector(neg_pcs)
    verify_c_covariance_vector(c_n)
    log_debug("\tClass +ve Bayesian Cov: ", c_p, ", Class -ve Bayesian Cov: ", c_n)
    write_file_ndarray("(12) cp [{}]".format(POSITIVE_CLASS), c_p)
    write_file_ndarray("(14) cn [{}]".format(NEGATIVE_CLASS), c_n)

    xp_p_pdf = get_bayes_2d_pdf(mu_p, c_p, pos_pcs, xp_pc, "XP{}".format(XP_INDEX))
    xp_n_pdf = get_bayes_2d_pdf(mu_n, c_n, neg_pcs, xp_pc, "XP{}".format(XP_INDEX))

    xn_p_pdf = get_bayes_2d_pdf(mu_p, c_p, pos_pcs, xn_pc, "XN{}".format(XN_INDEX))
    xn_n_pdf = get_bayes_2d_pdf(mu_n, c_n, neg_pcs, xn_pc, "XN{}".format(XN_INDEX))

    xp_prob_bayes = xp_p_pdf / (xp_p_pdf + xp_n_pdf)
    xn_prob_bayes = xn_n_pdf / (xn_p_pdf + xn_n_pdf)
    log("\tXP : ", XP_INDEX, ", truth: ", labels[XP_INDEX], ", Prob Histo {}: {}".format(
        POSITIVE_CLASS, xp_prob_bayes))
    write_file_array("(90) Result of classifying xp using Bayesian: ",
                     [labels[XP_INDEX], xp_prob_bayes])
    log("\tXN : ", XN_INDEX, ", truth: ", labels[XN_INDEX], ", Prob Histo {}: {}".format(
        NEGATIVE_CLASS, xn_prob_bayes))
    write_file_array("(94) Result of classifying xn using Bayesian: ",
                     [labels[XP_INDEX], xn_prob_bayes])


def get_xi_prob_from_sk(x_all, xp_index, xn_index):
    log("\n\n\n")
    log("SK")
    log("-----")
    skpca = sk_pca(n_components=2)
    pcs = skpca.fit_transform(x_all)
    log_debug("\tSK PCA: ", pcs.shape)

    pos_pcs = [pcs[idx] for idx in range(len(labels)) if labels[idx] == POSITIVE_CLASS]
    neg_pcs = [pcs[idx] for idx in range(len(labels)) if labels[idx] == NEGATIVE_CLASS]
    assert(len(pos_pcs) == len(pos_class_pcs))
    assert(len(neg_pcs) == len(neg_class_pcs))
    draw_scatter_plot(pcs[:, 0], P[:, 1], pcs[xp_index], pcs[xn_index], labels)
    get_xi_prob_by_bayes(pos_pcs, neg_pcs, pcs[xp_index], pcs[xn_index])


def get_training_accuracy_by_histo(hp, hn, mu_vec, v_vec, x_all, x_labels):
    num_right_predictions = 0
    xi_pred = [POSITIVE_CLASS]
    log("Xi\tPred?\tTruth?")
    for xi_index in range(len(x_all)):
        xi_truth = x_labels[xi_index]
        xi_p_n = get_prob_by_histo(hp, hn, x_all[xi_index],
                                   mu_vec, v_vec, "Histo Xi{}".format(xi_index))

        if xi_truth == POSITIVE_CLASS:
            if xi_p_n[0] > xi_p_n[1]:
                xi_pred = [POSITIVE_CLASS]
            else:
                xi_pred = [NEGATIVE_CLASS]
        else:
            if xi_p_n[1] > xi_p_n[0]:
                xi_pred = [NEGATIVE_CLASS]
            else:
                xi_pred = [POSITIVE_CLASS]

        if xi_pred == xi_truth:
            num_right_predictions += 1
            log_debug(xi_index, "\t", xi_pred, "\t", x_labels[xi_index])
        else:
            log_debug(xi_index, "\t", xi_pred, "\t", x_labels[xi_index])
    histo_accuracy = (num_right_predictions / (1.0 * len(x_labels))) * 100
    log("Histo Accuracy: ", histo_accuracy, ", TP+TN: ",
        num_right_predictions, ", Total: ", len(x_labels))
    write_file_array("(97) Training accuracy attained using histograms: ", [histo_accuracy])


def get_training_accuracy_by_bayes(pos_pcs, neg_pcs, p_all, x_labels):
    mu_p = get_mu_mean_vectors(pos_pcs)
    mu_n = get_mu_mean_vectors(neg_pcs)
    c_p = get_c_covariance_vector(pos_pcs)
    c_n = get_c_covariance_vector(neg_pcs)

    num_right_predictions = 0
    xi_pred = [POSITIVE_CLASS]
    p_all_major = p_all[:, 0:2]
    log("Xi\tPred?\tTruth?")
    for xi_index in range(len(p_all)):
        xi_pc = p_all_major[xi_index]
        xi_truth = x_labels[xi_index]
        xi_p_pdf = get_bayes_2d_pdf(mu_p, c_p, pos_pcs, xi_pc, "XP{}".format(XP_INDEX))
        xi_n_pdf = get_bayes_2d_pdf(mu_n, c_n, neg_pcs, xi_pc, "XP{}".format(XP_INDEX))
        xi_p_bayes = xi_p_pdf / (xi_p_pdf + xi_n_pdf)
        xi_n_bayes = xi_n_pdf / (xi_p_pdf + xi_n_pdf)

        if xi_truth == POSITIVE_CLASS:
            if xi_p_bayes > xi_n_bayes:
                xi_pred = [POSITIVE_CLASS]
            else:
                xi_pred = [NEGATIVE_CLASS]
        else:
            if xi_n_bayes > xi_p_bayes:
                xi_pred = [NEGATIVE_CLASS]
            else:
                xi_pred = [POSITIVE_CLASS]

        if xi_pred == xi_truth:
            num_right_predictions += 1
            log_debug(xi_index, "\t", xi_pred, "\t", x_labels[xi_index])
        else:
            log_debug(xi_index, "\t", xi_pred, "\t", x_labels[xi_index])
    bayesian_accuracy = (num_right_predictions / (1.0 * len(x_labels))) * 100
    log("Bayes Accuracy: ", bayesian_accuracy)
    write_file_array("(98) Training accuracy attained using Bayesian: ", [bayesian_accuracy])


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


def get_prinicipal_features_and_labels():
    x_, mu_, z_, c_, v_, p_, labels_ = get_pca()
    return p_[:, 0], p_[:, 1], labels_

