# HPL processed data
import sys, os
import numpy as np

import torch.utils.data as data


__all__ = ['FlyingThings3DSubset']

class FlyingThings3DSubset_labeled(data.Dataset):
    """
    Load the FlyingThings3DSubset dataset, and use the supered parameter to control the number of labeled data.
    Args:
        train (bool): training dataset(true), otherwise(false)
        transform(callable)
        full(bool): wether use the full dataset
        supered(int): 1/supered of the full data
    """
    def __init__(self, train, tranform, num_points, data_root, full=True, supered=0):
        super().__init__()
        self.root = os.path.join(data_root, 'FlyingThings3D_subset_processed_35m')
        self.train = train
        self.tranform = tranform
        self.num_points = num_points

        self.samples = self.make_dataset(full, supered)

        if len(self.samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int):
        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.tranform([pc1_loaded, pc2_loaded])

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

    def make_dataset(self, full, supered):
        root = os.path.realpath(os.path.expanduser(self.root))
        root = os.path.join(root, 'train') if self.train else os.path.join(root, 'val')

        all_paths = os.walk(root)
        useful_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])

        try:
            if self.train:
                assert(len(useful_paths) == 19639)
            else:
                assert(len(useful_paths) == 3824)
        except AssertionError:
            print(f'len(useful_paths) assert error: {len(useful_paths)}')
            sys.exit(1)

        if not full:
            res_paths = useful_paths[::4]
        else:
            res_paths = useful_paths
        
        if supered != 0:
            res_paths = res_paths[0:int(len(res_paths) / supered)]
        else:
            res_paths = res_paths
        
        return res_paths

    
    def pc_loader(self, path):
        pc1 = np.load(os.path.join(path, 'pc1.npy'))
        pc2 = np.load(os.path.join(path, 'pc2.npy'))
        # multiply -1 only for subset datasets
        pc1[..., -1] *= -1
        pc2[..., -1] *= -1
        pc1[..., 0] *= -1
        pc2[..., 0] *= -1

        return pc1, pc2
