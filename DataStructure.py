import torch
import numpy as np
import pandas as pd


class DataStructure:
    def __init__(self):
        self._data = {}

    def __getattr__(self, name):
        if name not in self._data:
            self._data[name] = DataStructure()
        if type(self._data[name]) is DataStructure:
            return self._data[name]
        if type(self._data[name]) is torch.device:
            return self._data[name]
        if type(self._data[name]) is torch.Tensor:
            return self._data[name]#.clone()
        if type(self._data[name]) is np.ndarray:
            return self._data[name]#.copy()
        return self._data[name]

    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def to_dict(self, flat=False):
        output = {}
        if not flat:
            for key, value in self._data.items():
                if type(value) is DataStructure:
                    output[key] = value.to_dict()
                else:
                    output[key] = value
            return output
        
        for key, value in self._data.items():
            if type(value) is DataStructure:
                value = value.to_dict(flat=True)
                if type(value) is dict:
                    for k, v in value.items():
                        output[key + '.' + k] = v
            else:
                output[key] = value
                
        return output
    
    def from_dict(self, data):
        for key, value in data.items():
            if type(value) is dict:
                self._data[key] = DataStructure()
                self._data[key].from_dict(value)
            else:
                self._data[key] = value

    def to_device(self, device):
        for key, value in self._data.items():
            if type(value) is torch.Tensor:
                self._data[key] = value.to(device)
            elif type(value) is DataStructure:
                value.to_device(device)
        return self
    

    def __repr__(self):
        return repr(self._data)
    
    def filter(self, mask, tensor_only=False):
        for key, value in self._data.items():
            if type(value) in [torch.Tensor]:
                self._data[key] = value[mask]
            elif (type(value) in [pd.DataFrame, pd.Series, np.ndarray]):
                if tensor_only:
                    continue
                self._data[key] = value[mask.cpu().numpy() if type(mask) is torch.Tensor else mask]
            else:
                value.filter(mask, tensor_only=tensor_only)
        
    def map(self, function, tensor_only=True, all_dtypes=False, use_name=False, name = tuple([])):
        for key, value in self._data.items():
            name_=name + (key,)
            # print(key)
            if type(value) in [torch.Tensor, np.ndarray]:
                if tensor_only and type(value) == np.ndarray:
                    self._data[key] = value
                else:
                    self._data[key] = function(value, name_) if use_name else function(value)

            elif type(value) is DataStructure:
                value.map(function, tensor_only=tensor_only, all_dtypes=all_dtypes, use_name=use_name, name=name_)
            elif all_dtypes:
                try:
                    self._data[key] = function(value, name_) if use_name else function(value)
                except:
                    pass

    def apply(self, function, all_dtypes=False):
        result = DataStructure()
        for key, value in self._data.items():
            if type(value) in [torch.Tensor]:
                result._data[key] = function(value)
            if type(value) in [np.ndarray]:
                try:
                    result._data[key] = function(value)
                except:
                    pass
            elif type(value) is DataStructure:
                result._data[key] = value.apply(function, all_dtypes=all_dtypes)
            elif all_dtypes:
                # try:
                result._data[key] = function(value)
                # except:
                #     pass
        return result

    def __len__(self):
        for key, value in self._data.items():
            if type(value) in [torch.Tensor, np.ndarray]:
                return len(value)
            elif type(value) is DataStructure:
                length = len(value)
                if length is not None:
                    return length
        return None
        
    
    def size(self, dim=None):
        value = next(iter(self._data.values()))
        if type(value) in [torch.Tensor, DataStructure]:
            if dim is None:
                return value.size()
            return value.size(dim)
        elif type(value) is np.ndarray:
            if dim is None:
                return value.shape
            return value.shape[dim]
    
    def clone(self):
        new = DataStructure()
        for key, value in self._data.items():
            if type(value) in [torch.Tensor, DataStructure]:
                new._data[key] = value.clone()
            elif type(value) is np.ndarray:
                new._data[key] = value.copy()
            else:
                new._data[key] = value
        return new
    
    def get(self, key, collapse=False, batch=None, target=None):
        if target is None:
            result = self._data[key]
            if collapse:
                mask = self._data['mask']
                if batch is None:
                    return result[mask]#.to(self.device)
                
                result = result[batch]#.to(self.device)
                mask = mask[batch]
                return result[mask]
            
            else:
                if batch is None:
                    return result#.to(self.device)
                
                return result[batch]#.to(self.device)
            
        else:
            for k, v in self._data.items():
                if type(v) is not DataStructure:
                    continue
                if key in v._data:
                    source = k
            B = self._data[source].mapIdx
            A = self._data[target].mapIdx
            output = self._data[source]._data[key]

            if batch is not None:
                A, B, output = A[batch], B[batch], output[batch]
            
            C = (A[..., None] == B[..., None, :]).int().argmax(-1)
            output = output[torch.arange(len(self))[:, None], C]

            if not collapse:
                return output
            
            mask = self._data[target].mask
            if batch is not None:
                mask = mask[batch]
            return output[mask]
        
    def fill(self, source, target):
        for key, value in self._data[source]._data.items():
            if key not in self._data[target]._data:
                self._data[target]._data[key] = self.get(key, target=target)

    def cat(self, other, dim=0, name=tuple([]), remove_missing=False, skip_errors=False):
        missing = []
        for key, value in self._data.items():
            name_ = name + (key,)
            if key not in other._data:
                missing.append(key)
                continue
            try:
                if type(value) is DataStructure:
                    self._data[key].cat(other._data[key], dim, name_, remove_missing, skip_errors)
                if type(value) is torch.Tensor:
                    self._data[key] = torch.cat([value, other._data[key].to(value.dtype)], dim)
                if type(value) is np.ndarray:
                    self._data[key] = np.concatenate([value, other._data[key].astype(value.dtype)], dim)
            except Exception as e:
                if skip_errors:
                    missing.append(key)
                    continue
                raise


        if remove_missing:
            for key in missing:
                del self._data[key]

    def merge(self, other, override=False, name=tuple([])):
        for key, value in self._data.items():
            name_ = name + (key,)
            if (type(value) is DataStructure) and (key in other._data):
                self._data[key] = value.merge(other._data[key], override=override, name=name_)

        for key, value in other._data.items():
            name_ = name + (key,)
            if override or (key not in self._data):
                self._data[key] = value
        return self


    def detach(self):
        for key, value in self._data.items():
            if type(value) is DataStructure:
                self._data[key].detach()
            if type(value) is torch.Tensor:
                self._data[key] = value.detach()
                
class DataStructureIterator:
    def __init__(self, dataStructure, indices=[]):
        self.current = 0
        
        self.data = []
        for i in range(len(dataStructure)):
            self.data.append(dataStructure.apply(lambda x : x[i]))

            for x in indices:
                if len(x) == 3:
                    idx, val, k = x
                    idx_ = idx[i]
                else:
                    idx, val, k, mask = x
                    idx_ = idx[i][mask[i]]
                if type(val) == pd.DataFrame:
                    v = val.iloc[idx_]
                else:
                    v = val[idx_]

                self.data[i]._data[k] = v

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < len(self.data):
            val = self.current
            self.current += 1
            return self.data[val]
        else:
            self.current = 0
            raise StopIteration
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)