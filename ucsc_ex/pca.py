#!/usr/bin/python

import os
import struct
import matplotlib as plt
from array import array as pyarray
from pylab import *
from numpy import *
import numpy.linalg as la

POSITIVE_CLASS = 5
NEGATIVE_CLASS = 6
RANDOM_REP_INDEX = 100
XP_INDEX = 25
XN_INDEX = 50
SCATTER_PLOT_ALPHA = 0.25
ENABLE_IMAGE_SHOW = True

np.set_printoptions(precision=4)
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
    print("Image!")
    if not ENABLE_IMAGE_SHOW:
        return
    plt.imshow(byte_array.reshape(28, 28), interpolation='None', cmap=cm.gray)
    plt.title(img_title)
    show()


def draw_image_subplot(byte_array, img_title, sub_plt):
    if not ENABLE_IMAGE_SHOW:
        return
    sub_plt.imshow(byte_array.reshape(28, 28), interpolation='None', cmap=cm.gray)
    sub_plt.set_title(img_title)
    sub_plt.axis('off')


def draw_scatter_plot(cloud_a, cloud_b, all_labels):
    print("Scatter Plot!")
    if not ENABLE_IMAGE_SHOW:
        return
    fig = plt.figure()
    cols = np.zeros((alen(all_labels), 4))
    for idx, ll in enumerate(all_labels):
        if ll == POSITIVE_CLASS:
            cols[idx] = [1, 0, 0, SCATTER_PLOT_ALPHA]
        if ll == NEGATIVE_CLASS:
            cols[idx] = [0, 1, 0, SCATTER_PLOT_ALPHA]
    random_order = np.arange(np.alen(all_labels))
    ax = fig.add_subplot(111, facecolor='black')
    ax.scatter(cloud_b, cloud_a, s=5, linewidths=0, facecolors=cols[random_order, :],
               marker="o")
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.title('scatter plot of Principal components')
    plt.show()


# converting from NX28X28 array into NX784 array
def get_x_feature_vectors(images_3d_array):
    flat_images = list()
    for i in images_3d_array:
        flat_images.append(i.ravel())
    x_feature_vectors = np.asarray(flat_images)
    print("X shape: ", x_feature_vectors.shape)
    print("X min/max: ", np.amin(x_feature_vectors), np.amax(x_feature_vectors))
    return x_feature_vectors


def get_mu_mean_vectors(x_feature_vector):
    mu = np.mean(x_feature_vector, axis=0)
    print("MU shape:", mu.shape)
    return mu


def get_z_variance_vector(x_feature_vector, mu_mean_vector):
    z = x_feature_vector - mu_mean_vector
    print("Z shape: ", z.shape)
    return z


def get_c_covariance_vector(z_variance_vector):
    c = np.cov(z_variance_vector, rowvar=False, ddof=1)
    print("C shape: ", c.shape)
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
    print("C Verification: Good!")
    return


def get_eigen_value_vector(c_covariance_vector):
    [eig_val, eig_vec] = la.eigh(c_covariance_vector)
    print("EigVal len: ", len(eig_val))
    print("EigVec shape: ", eig_vec.shape)
    return [eig_val, eig_vec]


def is_eigen_values_row_aligned(c_covariance_matrix, eig_val, eig_vec, row, col):
    # print(np.dot(C, row) / (EigVal[0] * row))
    # print(np.dot(C, col) / (EigVal[0] * col))
    return False


def verify_pca_vector(p_pca_vector):
    print("P shape: ", p_pca_vector.shape)


def verify_images_and_plots(x_matrix, mu_vec, z_vec, c_vec, v_eig_vector, p_pca_vector,
                            all_labels, image_index):
    print("Verify Images!")
    if not ENABLE_IMAGE_SHOW:
        return

    fig = plt.figure()
    draw_image_subplot(x_matrix[image_index], "X img {}".format(image_index), fig.add_subplot(241))

    print(v_eig_vector[0])
    print(v_eig_vector[1])

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


def get_pca():
    images, l_labels = load_mnist('training', digits=[POSITIVE_CLASS, NEGATIVE_CLASS])

    x = get_x_feature_vectors(images)
    mu = get_mu_mean_vectors(x)
    z = get_z_variance_vector(x, mu)
    c = get_c_covariance_vector(z)
    verify_c_covariance_vector(c)

    [eig_val, v] = get_eigen_value_vector(c)
    eig_row = v[0, :]
    eig_col = v[:, 0]

    if is_eigen_values_row_aligned(c, eig_val, v, eig_row, eig_col):
        print("")
    else:
        eig_val = np.flipud(eig_val)
        v = np.flipud(v.T)

    p = np.dot(z, v.T)
    verify_pca_vector(p)
    draw_scatter_plot(p[:, 0], p[:, 1], l_labels)
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


def get_histo_matrix(pos_class_pc1, pos_class_pc2, bin_counts):
    histo = np.zeros((bin_counts, bin_counts))
    for h, s in zip(pos_class_pc1, pos_class_pc2):
        rc_tup = get_histo_matrix_row_col(h, s)
        histo[rc_tup[0]][rc_tup[1]] = histo[rc_tup[0]][rc_tup[1]] + 1
    return histo


def get_mean_vector(vec):
    return np.mean(vec, axis=0, dtype=np.float64)


def get_covariance_matrix(vec):
    return np.cov(vec, rowvar=False, ddof=1)


def get_2d_pdf(xi, mu_x, cov_x):
    xi_mu = np.array([np.subtract(xi, mu_x)])
    cov_inverse = inv(cov_x)
    xi_mu_transpose = np.transpose(xi_mu)
    xi_scalar = (np.dot(xi_mu, cov_inverse).dot(xi_mu_transpose))
    part_2 = math.exp(-1 * xi_scalar / 2)
    cov_determinant = det(cov_x)
    part_1 = 1 / (2 * math.pi * math.pow(cov_determinant, 1 / 2))
    return part_1 * part_2


def get_post_probability_xi(xi, len_pos, mu_pos, cov_pos, len_neg, mu_neg, cov_neg):
    prior_prob_xi_given_pos = get_2d_pdf(xi, mu_pos, cov_pos)
    prior_prob_xi_given_pos_n = \
        len_pos * prior_prob_xi_given_pos

    prior_prob_xi_given_neg = get_2d_pdf(xi, mu_neg, cov_neg)
    prior_prob_xi_given_neg_n = \
        len_neg * prior_prob_xi_given_neg

    post_prob_neg_given_xi = prior_prob_xi_given_pos_n / (
        prior_prob_xi_given_pos_n + prior_prob_xi_given_neg_n)
    return post_prob_neg_given_xi


def print_samples():
    print("\nSample Size:")
    print("Class +ve({}): {}".format(POSITIVE_CLASS, len(pos_class_pc_1)))
    print("Class -ve({}): {}".format(NEGATIVE_CLASS, len(neg_class_pc_1)))
    print("Min pc1 {}, Max pc1 {}".format(min_pc_1, max_pc_1))
    print("Min pc2 {}, Max pc2 {}".format(min_pc_2, max_pc_2))
    print("Optimal bin count: {}".format(bin_count))


def print_histos():
    print(pos_class_histo)
    print(neg_class_histo)


def get_prob_by_histo(xi_vec, mu_vec, v_vec, p_vec, msg):
    print(msg)
    zi_vec = get_z_variance_vector(xi_vec, mu_vec)
    vi_2d_vec = v_vec[0:2, :]
    pi_vec = np.dot(zi_vec, vi_2d_vec.T)
    print("\t", pi_vec)
    r = get_bin(pi_vec[0], min(p_vec[0]), max(p_vec[0]))
    c = get_bin(pi_vec[1], min(p_vec[1]), max(p_vec[1]))
    print("\tR:", r, ", C:", c)


[X, MU, Z, C, V, P, labels] = get_pca()
[pc_1, pc_2, target_class] = get_pc_with_labels(P, labels)
# verify_images_and_plots(X, MU, Z, C, V, P, labels, RANDOM_REP_INDEX)
verify_images_and_plots(X, MU, Z, C, V, P, labels, XP_INDEX)
verify_images_and_plots(X, MU, Z, C, V, P, labels, XN_INDEX)

pos_class_pc_1 = [pc_1[idx] for idx in range(len(labels)) if labels[idx] == POSITIVE_CLASS]
pos_class_pc_2 = [pc_2[idx] for idx in range(len(labels)) if labels[idx] == POSITIVE_CLASS]
neg_class_pc_1 = [pc_1[idx] for idx in range(len(labels)) if labels[idx] == NEGATIVE_CLASS]
neg_class_pc_2 = [pc_2[idx] for idx in range(len(labels)) if labels[idx] == NEGATIVE_CLASS]

min_pc_1 = min(min(pos_class_pc_1), min(neg_class_pc_1))
max_pc_1 = max(max(pos_class_pc_1), max(neg_class_pc_1))
min_pc_2 = min(min(pos_class_pc_2), min(neg_class_pc_2))
max_pc_2 = max(max(pos_class_pc_2), max(neg_class_pc_2))

bin_count = get_optimal_bin_count(min(len(pos_class_pc_1),
                                      len(neg_class_pc_1)))

print_samples()

print("\nClass +ve Histo:")
pos_class_histo = get_histo_matrix(pos_class_pc_1, pos_class_pc_2, bin_count)

# print("\nPositive Class Histo:")
# print(np.histogram2d(pos_class_pc_1, pos_class_pc_2, bin_count)[0])

print("\nClass -ve Histo:")
neg_class_histo = get_histo_matrix(neg_class_pc_1, neg_class_pc_2, bin_count)

pos_class_pcs = list(zip(pos_class_pc_1, pos_class_pc_2))
neg_class_pcs = list(zip(neg_class_pc_1, neg_class_pc_2))

mu_pos_class = get_mean_vector(pos_class_pcs)
mu_neg_class = get_mean_vector(neg_class_pcs)
print("Class +ve Bayesian Mu: ", mu_pos_class, ", Class -ve Bayesian Mu: ", mu_neg_class)

cov_pos_class = get_covariance_matrix(pos_class_pcs)
cov_neg_class = get_covariance_matrix(neg_class_pcs)
print("Class +ve Bayesian Cov: ", cov_pos_class, ", Class -ve Bayesian Cov: ", cov_neg_class)

XP = X[XP_INDEX]
XN = X[XN_INDEX]
get_prob_by_histo(XP, MU, V, P, "XP{}".format(XP_INDEX))
