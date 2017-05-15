#!/usr/bin/python

import os
import struct
import matplotlib as plt
from array import array as pyarray
from pylab import *
from numpy import *
import numpy.linalg as la


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


def draw_image(byte_array):
    print("Image!")
    plt.imshow(byte_array.reshape(28, 28), interpolation='None', cmap=cm.gray)
    show()


def draw_scatter_plot(cloud_a, cloud_b):
    fig = plt.figure()
    cols = np.zeros((alen(labels), 4))
    for idx, ll in enumerate(labels):
        if ll == 5:
            cols[idx] = [1, 0, 0, 0.25]
        if ll == 6:
            cols[idx] = [0, 1, 0, 0.25]
    random_order = np.arange(np.alen(labels))
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


def verify_images_and_plots(p_pca_vector, v_eig_vector):
    recv = np.dot(p_pca_vector, v_eig_vector)
    draw_image(recv[25])

    draw_scatter_plot(P[:, 0], P[:, 1])

    x_rec1 = (np.dot(P[:, 0:1], V[0:1, :])) + MU
    draw_image(x_rec1[25])

    x_rec2 = (np.dot(P[:, 0:2], V[0:2, :])) + MU
    draw_image(x_rec2[25])

    x_rec = (np.dot(P, V)) + MU
    draw_image(x_rec[25])


images, labels = load_mnist('training', digits=[5, 6])
X = get_x_feature_vectors(images)
draw_image(X[25])

MU = get_mu_mean_vectors(X)
Z = get_z_variance_vector(X, MU)

C = get_c_covariance_vector(Z)
verify_c_covariance_vector(C)

[EigVal, V] = get_eigen_value_vector(C)
EigRow = V[0, :]
EigCol = V[:, 0]

if is_eigen_values_row_aligned(C, EigVal, V, EigRow, EigCol):
    print("")
else:
    EigVal = np.flipud(EigVal)
    V = np.flipud(V.T)

P = np.dot(Z, V.T)
verify_pca_vector(P)

verify_images_and_plots(P, V)

# get_positive_class(P, labels)
# get_negative_class(P, labels)

print()

