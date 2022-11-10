import os,glob
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import json


def metadata_reader(data_path):

    info = []
    data = sitk.ReadImage(data_path)
    # print(data)
    size = list(data.GetSize()[:2])
    z_size = data.GetSize()[-1]
    thick_ness = data.GetSpacing()[-1]
    pixel_spacing = list(data.GetSpacing()[:2])
    info.append(size)
    info.append(z_size)
    info.append(thick_ness)
    info.append(pixel_spacing)
    return info

# Different samples are saved in different folder
def get_metadata(input_path, save_path, image_postfix='data.nii.gz'):

    id_list = os.listdir(input_path)
    info = []
    for ID in tqdm(id_list):
        info_item = [ID]
        data_path = os.path.join(input_path, ID)
        image_path = glob.glob(os.path.join(data_path, image_postfix))[0]
        info_item.extend(metadata_reader(image_path))
        info.append(info_item)
    col = ['id', 'size', 'num', 'thickness', 'pixel_spacing']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(save_path, index=False)


if __name__ == "__main__":

    # HaN_GTV
    json_file = './static_files/HaN_GTV.json'
  
    # THOR_GTV
    # json_file = './static_files/THOR_GTV.json'

    with open(json_file, 'r') as fp:
        info = json.load(fp)
    get_metadata(info['nii_path'], info['metadata_path'],image_postfix=info['image_postfix'])
    
