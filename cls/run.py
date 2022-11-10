import os
import argparse
from trainer import Slice_Classifier
import pandas as pd
from config import INIT_TRAINER, SETUP_TRAINER, LABEL_DICT,CURRENT_FOLD, FOLD_NUM, WEIGHT_PATH_LIST
from config import VERSION, DISEASE, ROI_NAME, TEST_LABEL_DICT

import time
import numpy as np
import random

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def get_cross_validation_by_sample(path_list, fold_num, current_fold):

    sample_list = list(set([os.path.basename(case).split('_')[0] for case in path_list]))
    sample_list.sort()
    print('number of sample:',len(sample_list))

    _len_ = len(sample_list) // fold_num

    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(sample_list[start_index:])
        train_id.extend(sample_list[:start_index])
    else:
        validation_id.extend(sample_list[start_index:end_index])
        train_id.extend(sample_list[:start_index])
        train_id.extend(sample_list[end_index:])

    train_path = []
    validation_path = []
    for case in path_list:
        if os.path.basename(case).split('_')[0] in train_id:
            train_path.append(case)
        else:
            validation_path.append(case)

    random.shuffle(train_path)
    random.shuffle(validation_path)
    print("Train set length:", len(train_path),
          "\nVal set length:", len(validation_path))
    return train_path, validation_path


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train-cross', choices=["train-cross","train", "test", "test-cross"],
                        help='choose the mode', type=str)
    parser.add_argument('-s', '--save', default='no', choices=['no', 'n', 'yes', 'y'],
                        help='save the forward middle features or not', type=str)
    args = parser.parse_args()
    

    label_dict = LABEL_DICT
    path_list = list(label_dict.keys())

    if args.mode != 'test-cross' and args.mode != 'train-cross':
        classifier = Slice_Classifier(**INIT_TRAINER)
        print(get_parameter_number(classifier.net))

    # Training with cross validation
    ###############################################
    if args.mode == 'train-cross':
        print("dataset length is %d"%len(path_list))
        for current_fold in range(1, FOLD_NUM+1):
            print("=== Training Fold ", current_fold, " ===")
            classifier = Slice_Classifier(**INIT_TRAINER)

            train_path, val_path = get_cross_validation_by_sample(
                path_list, FOLD_NUM, current_fold)

            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['label_dict'] = label_dict
            SETUP_TRAINER['cur_fold'] = current_fold

            start_time = time.time()
            classifier.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time()-start_time))

    ###############################################

    # Training
    ###############################################
    elif args.mode == 'train':

        print("dataset length is %d"%len(path_list))

        train_path, val_path = get_cross_validation_by_sample(
            path_list, FOLD_NUM, CURRENT_FOLD)
        SETUP_TRAINER['train_path'] = train_path
        SETUP_TRAINER['val_path'] = val_path
        SETUP_TRAINER['label_dict'] = label_dict
        SETUP_TRAINER['cur_fold'] = CURRENT_FOLD

        start_time = time.time()
        classifier.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time()-start_time))
    ###############################################

    # Testing
    ###############################################
    elif args.mode == 'test':
        test_label_dict = TEST_LABEL_DICT
        test_path = list(test_label_dict.keys())
        print('test set length:%d'%len(test_path))
        save_path = './analysis/result/{}/{}'.format(DISEASE,VERSION)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        csv_path = os.path.join(save_path, f'{ROI_NAME}_fold{CURRENT_FOLD}.csv')

        start_time = time.time()
        if args.save == 'no' or args.save == 'n':
            result, _, _ = classifier.test(test_path, test_label_dict)
            print('run time:%.4f' % (time.time()-start_time))
        else:
            result, feature_in, feature_out = classifier.test(
                test_path, test_label_dict, hook_fn_forward=True)
            print('run time:%.4f' % (time.time()-start_time))
            # save the avgpool output
            print(feature_in.shape, feature_out.shape)
            feature_dir = './analysis/mid_feature/{}/{}/{}'.format(DISEASE,VERSION,ROI_NAME)
            if not os.path.exists(feature_dir):
                os.makedirs(feature_dir)
            from utils import save_as_hdf5
            for i in range(len(test_path)):
                name = os.path.basename(test_path[i])
                feature_path = os.path.join(feature_dir, name)
                save_as_hdf5(feature_in[i], feature_path, 'feature_in')
                save_as_hdf5(feature_out[i], feature_path, 'feature_out')
        info = {}
        info['id'] = test_path
        info['label'] = result['true']
        info['pred'] = result['pred']
        info['prob'] = result['prob']
        # print(classification_report(result['true'], list(np.array(result['prob']) > 0.7), target_names=['without','with'],output_dict=False))
        print(classification_report(result['true'],result['pred'], target_names=['without','with'],output_dict=False))
        print(confusion_matrix(result['true'], result['pred']))
        csv_file = pd.DataFrame(info)
        csv_file.to_csv(csv_path, index=False)
    ###############################################


    # test with cross validation
    ###############################################
    elif args.mode == 'test-cross':
        test_label_dict = TEST_LABEL_DICT
        test_path = list(test_label_dict.keys())
        print('test set length:%d'%len(test_path))
        save_path = './analysis/result/{}/{}'.format(DISEASE,VERSION)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        start_time = time.time()
        csv_path = os.path.join(save_path,ROI_NAME + '_vote.csv')

        all_result = {
            'id':[],
            'true': [],
            'pred': []
        }

        start_time = time.time()
        print(WEIGHT_PATH_LIST)
        for i, weight_path in enumerate(WEIGHT_PATH_LIST):
            print("Inference %d fold..." % (i+1))
            INIT_TRAINER['weight_path'] = weight_path
            classifier = Slice_Classifier(**INIT_TRAINER)

            result, _, _ = classifier.test(test_path, test_label_dict)
            all_result['pred'].append(result['pred'])
            # print(np.array(all_result['pred']).shape)
            
        all_result['true'] = result['true']
        all_result['pred'] = list((np.sum(np.array(all_result['pred']),axis=0) > len(WEIGHT_PATH_LIST)//2).astype(np.int8))
        # print(len(all_result['pred']))
        print(classification_report(result['true'],all_result['pred'], target_names=['without','with'],output_dict=False))
        print(confusion_matrix(result['true'],all_result['pred']))
        all_result['id'] = test_path
        csv_file = pd.DataFrame(all_result)
        csv_file.to_csv(csv_path, index=False)
    ###############################################


