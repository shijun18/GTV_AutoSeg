import json
from utils import get_weight_path,get_weight_list,csv_reader_single,csv_reader_single_ratio


__net__ = ['resnet18','resnet34', 'resnet50','swin_transformer']

__disease__ = ['Cervical','Nasopharynx','Structseg_HaN','Structseg_THOR','SegTHOR']

json_path = {
    'HaN_GTV':'../converter/nii_converter/static_files/HaN_GTV.json',
    'THOR_GTV':'../converter/nii_converter/static_files/THOR_GTV.json',
}


DISEASE = 'HaN_GTV' 
NET_NAME = 'resnet18'
VERSION = 'v1.0-roi-all'

DEVICE = '3'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 1,2,3,4
CURRENT_FOLD = 5
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5
LABEL_DICT = {}

with open(json_path[DISEASE], 'r') as fp:
    info = json.load(fp)

# Arguments for trainer initialization
#--------------------------------- single or multiple
ROI_NUMBER = 2 # or 1,2,...
NUM_CLASSES = info['annotation_num'] + 1# 2 for binary
if ROI_NUMBER is not None:
    NUM_CLASSES = 2
    ROI_NAME = info['annotation_list'][ROI_NUMBER - 1]
    # ratio
    # all
    if 'all' in VERSION:
        LABEL_DICT = csv_reader_single(info['2d_data']['csv_path'],'path',ROI_NAME)
    elif 'equal' in VERSION:
        LABEL_DICT = csv_reader_single_ratio(info['2d_data']['csv_path'],'path',ROI_NAME,ratio=1.0,reversed_flag=False)
    elif 'half' in VERSION:
        LABEL_DICT = csv_reader_single_ratio(info['2d_data']['csv_path'],'path',ROI_NAME,ratio=0.5,reversed_flag=False)
    elif 'quar' in VERSION:
        LABEL_DICT = csv_reader_single_ratio(info['2d_data']['csv_path'],'path',ROI_NAME,ratio=0.25,reversed_flag=False)


SCALE = info['scale'][ROI_NAME]
#---------------------------------

#--------------------------------- mode and data path setting
CKPT_PATH = './ckpt/{}/{}/{}/fold{}'.format(DISEASE,VERSION,ROI_NAME,str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

if PRE_TRAINED:
  WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/{}/{}/'.format(DISEASE,VERSION,ROI_NAME))
else:
  WEIGHT_PATH_LIST = None
#---------------------------------


#--------------------------------- others
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 2e-3,
    'n_epoch': 120,
    'channels': 1,
    'num_classes': NUM_CLASSES,
    'scale':SCALE,
    'input_shape': (512, 512),
    'crop': 0,
    'batch_size': 128,
    'num_workers': max(8,GPU_NUM*4),
    'device': DEVICE,
    'pre_trained': PRE_TRAINED,
    'weight_path': WEIGHT_PATH,
    'weight_decay': 0.0001,
    'momentum': 0.99,
    'mean': None,
    'std': None,
    'gamma': 0.1,
    'milestones': [110],
    'use_fp16':True,
    'transform':[1,2,3,7,9,18,10,16] if 'roi' not in VERSION else [1,19,2,3,7,9,18,10,16], # [1,2,3,7,9,10,16]
    'drop_rate':0.2, #0.5
    'external_pretrained':True if 'pretrained' in VERSION else False,#False
    'use_mixup':True if 'mixup' in VERSION else False,
    'use_cutmix':True if 'cutmix' in VERSION else False,
    'mix_only': True if 'only' in VERSION else False
}
#---------------------------------

# Arguments when perform the trainer
__loss__ = ['Cross_Entropy','TopkCrossEntropy','SoftCrossEntropy','F1_Loss','TopkSoftCrossEntropy','DynamicTopkCrossEntropy','DynamicTopkSoftCrossEntropy']

loss_index = eval(VERSION.split('.')[-1].split('-')[0])
LOSS_FUN = __loss__[loss_index]
print('>>>>> loss fun:%s'%LOSS_FUN)

SETUP_TRAINER = {
    'output_dir': './ckpt/{}/{}/{}'.format(DISEASE,VERSION,ROI_NAME),
    'log_dir': './log/{}/{}/{}'.format(DISEASE,VERSION,ROI_NAME),
    'optimizer': 'AdamW',
    'loss_fun': LOSS_FUN,
    'class_weight': None,
    'lr_scheduler': 'CosineAnnealingWarmRestarts', #'MultiStepLR','CosineAnnealingWarmRestarts' for fine-tune and warmup
    'monitor':'val_f1'
}
#---------------------------------

TEST_LABEL_DICT = {}
if DISEASE in ['HaN_GTV','THOR_GTV']:
    if ROI_NUMBER is not None:
       TEST_LABEL_DICT = csv_reader_single(info['2d_data']['test_csv_path'],'path',ROI_NAME)