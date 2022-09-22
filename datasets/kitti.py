import os
import numpy as np

import torch.utils.data as data

__all__ = ['KITTI']

class KITTI(data.Dataset):
    """
    Load the KITTI Scene Flow 2015 dataset, and use the supered parameter to control the number of labeled data.
    Args:
        train (bool): training dataset(true), otherwise(false)
        transform(callable)
        full(bool): wether use the full dataset
    """
    def __init__(self, train, transform, num_points, data_root, remove_ground = True):
        self.root = os.path.join(data_root, 'KITTI_processed_occ_final')
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = remove_ground
    
        self.samples = self.make_dataset()

        if len(self.samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])

        if pc1_transformed is None:
            print(f"path {self.samples[index]} is None.")
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)
        
        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed

        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index]

    def __repr__(self):
        '''The object name.(output of object <Dataset dataset_name...>)'''
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def make_dataset(self):
        do_mapping = True
        root = os.path.realpath(os.path.expanduser(self.root))
        
        all_paths = os.walk(root)
        useful_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])

        try:
            assert (len(useful_paths) == 200)
        except AssertionError:
            print('assert (len(useful_paths) == 200) failed!', len(useful_paths))

        if do_mapping:
            mappting_path = os.path.join(os.path.dirname(__file__), 'KITTI_mapping.txt')

            with open(mappting_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in useful_paths if lines[int(os.path.split(path)[-1])] != '']
        
        if self.train:
            res_path = useful_paths[0:100]
        else:
            res_path = useful_paths[100:]

        return res_path
    
    def pc_loader(self, path):
        pc1 = np.load(os.path.join(path, 'pc1.npy'))
        pc2 = np.load(os.path.join(path, 'pc2.npy')) 

        if self.remove_ground:
            is_ground = np.logical_and(pc1[:, 1] < -1.4 , pc2[:, 1] < -1.4)
            not_ground = np.logical_not(is_ground)

            pc1 = pc1[not_ground]
            pc2 = pc2[not_ground]
        
        return pc1, pc2