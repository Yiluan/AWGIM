from __future__ import print_function
import numpy as np
import pickle as pkl


class dataset_mini(object):
    def __init__(self, split, args):
        self.split = split
        self.seed = args.seed
        self.root_dir = 'data/miniImagenet'

    def load_data_pkl(self):
        pkl_name = '{}/{}_embeddings.pkl'.format(self.root_dir, self.split)
        f = open(pkl_name, 'rb')
        data = pkl.load(f, encoding='latin1')
        f.close()
        self.data = data
        self.n_classes = data.shape[0]
        print('labeled data:', np.shape(self.data), self.n_classes)

    def next_data(self, n_way, n_shot, n_query):
        support = np.zeros([n_way, n_shot, 640], dtype=np.float32)
        query = np.zeros([n_way, n_query, 640], dtype=np.float32)
        selected_classes = np.random.permutation(self.n_classes)[:n_way]
        for i, cls in enumerate(selected_classes):  # train way
            idx1 = np.random.permutation(600)[:n_shot + n_query]
            support[i] = self.data[cls, idx1[:n_shot]]
            query[i] = self.data[cls, idx1[n_shot:]]

        support_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_shot)).astype(np.uint8)
        query_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)

        query = np.reshape(query, (n_way * n_query, 640))
        query_labels = np.reshape(query_labels, (n_way * n_query))
        support = np.reshape(support, (n_way * n_shot, 640))
        support_labels = np.reshape(support_labels, (n_way * n_shot))
        return support, support_labels, query, query_labels

