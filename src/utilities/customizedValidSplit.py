import torch
import pandas as pd
import numpy as np
from math import ceil
from skorch.dataset import ValidSplit

class CustomizedValidSplit(ValidSplit):
    def __call__(self, dataset, y=None, groups=None):
        bad_y_error = ValueError(
            "Stratified CV requires explicitly passing a suitable y.")
        if (y is None) and self.stratified:
            raise bad_y_error

        cv = self.check_cv(y)
        if self.stratified and not self._is_stratified(cv):
            raise bad_y_error

        # pylint: disable=invalid-name
        len_dataset = len(dataset)
        if y is not None:
            len_y = len(y)
            if len_dataset != len_y:
                raise ValueError("Cannot perform a CV split if dataset and y "
                                 "have different lengths.")

        args = (np.arange(len_dataset),)
        if self._is_stratified(cv):
            idx_train, idx_valid = self._stratified_split_(y)
        else:
            idx_train, idx_valid = next(iter(cv.split(*args, groups=groups)))

        dataset_train = torch.utils.data.Subset(dataset, idx_train)
        dataset_valid = torch.utils.data.Subset(dataset, idx_valid)
        return dataset_train, dataset_valid
        # return dataset_valid, dataset_valid
    
    def _stratified_split_(self, y):
        """stratified KF TODO:
        """
        def sample_group(group, sampling_rate):
            sample_size = max(1, ceil((len(group) * sampling_rate)))
            return group.sample(n=sample_size)

        if self.cv < 1:
            sampling_rate = 1 - self.cv # sample train dataset
        elif self.cv > 1:
            sampling_rate = 1 - 1 / self.cv 
        else:
            raise ValueError("Not Implemented")
            
        y = pd.DataFrame(y,  columns=['y'])
        train_y = y.groupby('y', group_keys=False).apply(sample_group, sampling_rate=sampling_rate)
        idx_train = train_y.index
        idx_valid = y.index.difference(train_y.index)
        # train_y_u = set(train_y['y'].unique())
        # test_y_u = set(y.loc[idx_valid]['y'].unique())
        # print(train_y_u)
        # print(test_y_u)
        # print(train_y_u - test_y_u)
        
        # print(test_y_u - train_y_u)
        # print(idx_train.shape, idx_valid.shape)
        return idx_train.values, idx_valid.values
    

    