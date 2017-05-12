import os
import struct
from array import array as pyarray
from numpy import int8, uint8
from pylab import *
from numpy import *


def load_mnist(dataset="training", digits=range(10), path="./data/3/"):
    """
    Adapted from: http://cvxopt.org/applications/svm/index.html?highlight=mnist
    """
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx')
        fname_lbl = os.path.join(path, 't10k-labels-idx')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, lbl_size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()
    print("Labels: ", len(lbl))

    fimg = open(fname_img, 'rb')
    magic_nr, img_size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
    print("Images: ", len(img))

    ind = [k for k in range(len(img)) if lbl[k] in digits]
    N = len(ind)
    print("N: ", ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


images, labels = load_mnist('training', digits=[3,4], path="./data/3/")

# converting from NX28X28 array into NX784 array
flatimages = list()
for i in images:
    flatimages.append(i.ravel())
X = np.asarray(flatimages)

print("Check shape of matrix", X.shape)
print("Check Mins and Max Values",np.amin(X),np.amax(X))
print("\nCheck training vector by plotting image \n")
plt.imshow(X[20].reshape(28, 28), interpolation='None', cmap=cm.gray)
show()