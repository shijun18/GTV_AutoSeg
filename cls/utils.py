import pandas as pd
import numpy as np
import h5py
import os
import shutil
import random


def csv_reader_single_ratio(csv_file,key_col=None,value_col=None,ratio=0.5,reversed_flag=False):
    '''
    Extracts the specified single column, return a single level dict.
    The value of specified column as the key of dict.

    Args:
    - csv_file: file path
    - key_col: string, specified column as key, the value of the column must be unique. 
    - value_col: string,  specified column as value
    '''
    file_csv = pd.read_csv(csv_file)
    key_list = file_csv[key_col].values.tolist()
    value_list = file_csv[value_col].values.tolist()
    
    with_dict = {}
    without_dict = {}
    for key_item,value_item in zip(key_list,value_list):
        if value_item != 0:
            with_dict[key_item] = value_item
        else:
            without_dict[key_item] = value_item
    if reversed_flag:
        if int(len(without_dict)/ratio) < len(with_dict):
            with_list = list(with_dict.keys())
            random.shuffle(with_list)
            with_list = with_list[:int(len(without_dict)/ratio)]
            tmp_dict = {}
            for case in with_list:
                tmp_dict[case] = 1
            with_dict = tmp_dict
    else:
        if int(len(with_dict)/ratio) < len(without_dict):
            without_list = list(without_dict.keys())
            random.shuffle(without_list)
            without_list = without_list[:int(len(with_dict)/ratio)]
            tmp_dict = {}
            for case in without_list:
                tmp_dict[case] = 0
            without_dict = tmp_dict
    with_dict.update(without_dict)
    return with_dict


def make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def remove_dir(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)


def get_weight_list(ckpt_path):
    path_list = []
    for fold in os.scandir(ckpt_path):
        if fold.is_dir():
            weight_path = os.listdir(fold.path)
            weight_path.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            path_list.append(os.path.join(fold.path,weight_path[-1]))
            # print(os.path.join(fold.path,weight_path[-1]))
    
    return path_list



def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None


def remove_weight_path(ckpt_path,retain=5):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path,retain=5):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain)
        else:
            remove_weight_path(ckpt_path,retain)
            break    



def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()
    

def csv_reader_single(csv_file,key_col=None,value_col=None):
  '''
  Extracts the specified single column, return a single level dict.
  The value of specified column as the key of dict.

  Args:
  - csv_file: file path
  - key_col: string, specified column as key, the value of the column must be unique. 
  - value_col: string,  specified column as value
  '''
  file_csv = pd.read_csv(csv_file)
  key_list = file_csv[key_col].values.tolist()
  value_list = file_csv[value_col].values.tolist()
  
  target_dict = {}
  for key_item,value_item in zip(key_list,value_list):
    target_dict[key_item] = value_item

  return target_dict



def csv_reader_multiple(csv_file, key_col=None):
    '''
    Extracts multiple columns as value, return a single level dict.
    The value of multiple columns as the key of dict.

    Args:
    - csv_file: file path
    - key_col: string, specified column as key, the value of the column must be unique. 
    '''
    file_csv = pd.read_csv(csv_file)
    key_list = file_csv[key_col].values.tolist()
    del file_csv[key_col]
    values = np.array(file_csv.values)
    target_dict = {}
    for i in range(values.shape[0]):
        target_dict[key_list[i]] = list(values[i])
    
    return target_dict
