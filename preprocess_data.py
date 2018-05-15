import os
import sys
import numpy as np
import h5py
from tqdm import tqdm
# from scipy.spatial.distance import cdist
# import sklearn
# from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
# import time

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_FILES = getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
if not os.path.exists(DATA_DIR):
    print("Please run provider.py to download the data first!")
    exit(0)

def transform(pc, m):
    """
    Transform one set of point cloud (i.e. one shape).
    pc: (n,3) array
    return: (n,3*m) array, order by distance.
    """
    # print(pc.shape)
    n = pc.shape[0]
    knn = neighbors.NearestNeighbors(m, metric = "euclidean", algorithm = "ball_tree").fit(pc)
    nbors = knn.kneighbors(pc, return_distance=False) # sorted
    return pc[nbors].reshape((n,3*m))

def main(m = 8):
    """
    Transform data, from one point to m points, order by distance. And store it.
    params:
        m: how many points in a row, including the point itself, default is 8
    return:
        None
    """
    ALL_FILES = TRAIN_FILES + TEST_FILES
    print(ALL_FILES)
    for f in range(len(ALL_FILES)):
        # generate transformed data
        fname = ALL_FILES[f]
        print("Transforming", fname)
        current_data, current_label = loadDataFile(fname)
        b,n,_ = current_data.shape
        res = np.zeros((b,n,3*m))
        for i in tqdm(range(current_data.shape[0])):
            res[i] = transform(current_data[i], m)
        # store
        fname_new = f"{fname}.{m}.h5"
        with h5py.File(fname_new, 'w') as h5f:
            h5f.create_dataset('data',data=res)
            h5f.create_dataset('label',data=current_label)

def test(m = 8):
    """
    Test the main function.
    """
    DIR =  os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048")
    for filename in os.listdir(DIR):
        if filename.endswith(f"{m}.h5"):
            fullname = os.path.join(DIR, filename)
            print("Testing", fullname)
            current_data, current_label = loadDataFile(fullname)
            print(current_data.shape)
            print(current_label.shape)

if __name__ == "__main__":
    # main()
    test()
