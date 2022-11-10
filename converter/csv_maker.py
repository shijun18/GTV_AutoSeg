import sys
sys.path.append('..')
import os
import json
import pandas as pd 
import numpy as np
from skimage import measure

from converter.utils import hdf5_reader



def csv_maker(input_path,save_path,label_list):
    csv_info = []
    entry = os.scandir(input_path)
    len_ = len(os.listdir(input_path))
    count = 0 
    
    for item in entry:
        csv_item = []
        csv_item.append(item.path)
        tag_array = np.zeros((len(label_list) + 1,),dtype=np.uint8)
        label = hdf5_reader(item.path,'label')
        # print(np.unique(label).astype(np.uint8))
        tag_array[np.unique(label).astype(np.uint8)] = 1
        csv_item.extend(list(tag_array[1:]))
        # print(item.path)
        # print(list(tag_array[1:]))
        csv_info.append(csv_item)
        count += 1
        sys.stdout.write('\r Current: %d / %d'%(count,len_))
    print('\n')
    col = ['path'] + label_list
    csv_file = pd.DataFrame(columns=col, data=csv_info)
    csv_file.to_csv(save_path, index=False)


def area_compute(input_path,save_path,label_list):
    label_len = len(label_list)
    csv_info = []
    entry = os.scandir(input_path)
    len_ = len(os.listdir(input_path))
    count = 0 
    
    for item in entry:
        csv_item = []
        csv_item.append(item.path)
        area_list = list(np.zeros((label_len*2,),dtype=np.uint8))
        label = hdf5_reader(item.path,'label')
        if np.sum(label) !=0:
            for i in range(label_len):
                roi = (label==i+1).astype(np.uint8)
                area_list[i] = np.sum(roi)
                roi = measure.label(roi)
                area_list[i+label_len] = np.amax(roi)
                # area = []
                # for j in range(1,np.amax(roi) + 1):
                #     area.append(np.sum(roi == j))
                # area_list[i] = area
                
        csv_item.extend(area_list)

        csv_info.append(csv_item)
        count += 1
        sys.stdout.write('\r Current: %d / %d'%(count,len_))
    print('\n')
    col = ['path'] + label_list + [f'area_num_{str(i+1)}' for i in range(label_len)]
    csv_file = pd.DataFrame(columns=col, data=csv_info)
    csv_file.to_csv(save_path, index=False)


def volume_compute(input_path,save_path,label_list):
    label_len = len(label_list)
    csv_info = []
    entry = os.scandir(input_path)
    len_ = len(os.listdir(input_path))
    count = 0 
    
    for item in entry:
        csv_item = []
        csv_item.append(item.path)
        area_list = list(np.zeros((label_len*2,),dtype=np.uint8))
        label = hdf5_reader(item.path,'label')
        if np.sum(label) !=0:
            for i in range(label_len):
                roi = (label==i+1).astype(np.uint8)
                area_list[i] = np.sum(roi)
                roi = measure.label(roi)
                area_list[i+label_len] = np.amax(roi)
                # area = []
                # for j in range(1,np.amax(roi) + 1):
                #     area.append(np.sum(roi == j))
                # area_list[i] = area
                
        csv_item.extend(area_list)

        csv_info.append(csv_item)
        count += 1
        sys.stdout.write('\r Current: %d / %d'%(count,len_))
    print('\n')
    col = ['path'] + label_list + [f'area_num of {i}' for i in label_list]
    csv_file = pd.DataFrame(columns=col, data=csv_info)
    csv_file.to_csv(save_path, index=False)

def distribution_compute(input_path,save_path,label_list):
    label_len = len(label_list)
    csv_info = []
    entry = os.scandir(input_path)
    len_ = len(os.listdir(input_path))
    count = 0 
    
    for item in entry:
        csv_item = []
        csv_item.append(item.path)
        area_list = list(np.zeros((label_len*2,),dtype=np.uint8))
        label = hdf5_reader(item.path,'label')
        if np.sum(label) !=0:
            for i in range(label_len):
                roi = (label==i+1).astype(np.uint8)
                with_num = np.sum(np.sum(roi,axis=(1,2)) > 0)
                area_list[i] = with_num
                area_list[i+label_len] = label.shape[0] - with_num
                
        csv_item.extend(area_list)

        csv_info.append(csv_item)
        count += 1
        sys.stdout.write('\r Current: %d / %d'%(count,len_))
    print('\n')
    col = ['path'] + label_list + [f'without {i}' for i in label_list]
    csv_file = pd.DataFrame(columns=col, data=csv_info)
    csv_file.to_csv(save_path, index=False)


if __name__ == "__main__":

    # han_gtv
    json_file = './nii_converter/static_files/HaN_GTV.json'
    # thor_gtv
    # json_file = './nii_converter/static_files/THOR_GTV.json'
    
    with open(json_file, 'r') as fp:
        info = json.load(fp)
        input_path = info['2d_data']['save_path']
        save_path = info['2d_data']['csv_path']
        # area_path = info['2d_data']['area_path']
        

        # for test data
        # input_path = info['2d_data']['test_path']
        # save_path = info['2d_data']['test_csv_path']
        # area_path = info['2d_data']['test_area_path']

        # for 3d volume
        # input_path = info['npy_path']
        # volume_path = info['volume_csv']
        
        # for slice distribution
        # input_path = info['npy_path']
        # dis_path = info['distribution_csv']
        
    csv_maker(input_path,save_path,info['annotation_list'])
    # area_compute(input_path,area_path,info['annotation_list'])
    # volume_compute(input_path,volume_path,info['annotation_list'])
    # distribution_compute(input_path,dis_path,info['annotation_list'])
