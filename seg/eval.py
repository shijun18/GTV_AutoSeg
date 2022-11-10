import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score,precision_score
from torch.utils.data import DataLoader
from torchvision import transforms
from data_utils.data_loader import DataGenerator, To_Tensor, CropResize, Trunc_and_Normalize
from data_utils.transformer import Get_ROI
from torch.cuda.amp import autocast as autocast
from utils import get_weight_path,multi_dice,multi_hd,ensemble,post_seg,multi_vs,multi_jc
import warnings
from utils import csv_reader_single,hdf5_reader
warnings.filterwarnings('ignore')

def resize_and_pad(pred,true,num_classes,target_shape,bboxs):
    from skimage.transform import resize
    final_pred = []
    final_true = []

    for bbox, pred_item, true_item in zip(bboxs,pred,true):
        h,w = bbox[2]-bbox[0], bbox[3]-bbox[1]
        new_pred = np.zeros(target_shape,dtype=np.float32)
        new_true = np.zeros(target_shape,dtype=np.float32)
        for z in range(1,num_classes):
            roi_pred = resize((pred_item == z).astype(np.float32),(h,w),mode='constant')
            new_pred[bbox[0]:bbox[2],bbox[1]:bbox[3]][roi_pred>=0.5] = z
            roi_true = resize((true_item == z).astype(np.float32),(h,w),mode='constant')
            new_true[bbox[0]:bbox[2],bbox[1]:bbox[3]][roi_true>=0.5] = z
        final_pred.append(new_pred)
        final_true.append(new_true)
    
    final_pred = np.stack(final_pred,axis=0)
    final_true = np.stack(final_true,axis=0)
    return final_pred, final_true


def get_net(net_name,encoder_name,channels=1,num_classes=2,input_shape=(512,512),**kwargs):

    if net_name == 'unet':
        if encoder_name in ['simplenet','swin_transformer','swinplusr18']:
            from model.unet import unet
            net = unet(net_name,
                encoder_name=encoder_name,
                in_channels=channels,
                classes=num_classes,
                aux_classifier=True)
        else:
            import segmentation_models_pytorch as smp
            net = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )
    elif net_name == 'unet++':
        if encoder_name is None:
            raise ValueError(
                "encoder name must not be 'None'!"
            )
        else:
            import segmentation_models_pytorch as smp
            net = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )
    
    elif net_name == 'deeplabv3+':
        if encoder_name in ['swinplusr18']:
            from model.deeplabv3plus import deeplabv3plus
            net = deeplabv3plus(net_name,
                encoder_name=encoder_name,
                in_channels=channels,
                classes=num_classes)
        else:
            import segmentation_models_pytorch as smp
            net = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )
    elif net_name == 'res_unet':
        from model.res_unet import res_unet
        net = res_unet(net_name,
            encoder_name=encoder_name,
            in_channels=channels,
            classes=num_classes,
            **kwargs)

    elif net_name == 'sanet':
            from model.sanet import sanet
            net = sanet(net_name,
            encoder_name=encoder_name,
            in_channels=channels,
            classes=num_classes,
            **kwargs)
    
    elif net_name == 'att_unet':
        from model.att_unet import att_unet
        net = att_unet(net_name,
            encoder_name=encoder_name,
            in_channels=channels,
            classes=num_classes)
    
    
    elif net_name.startswith('vnet'):
        import model.vnet as vnet
        net = vnet.__dict__[net_name](
            init_depth=input_shape[0],
            in_channels=channels,
            classes=num_classes,
        )

    ## external transformer + U-like net
    elif net_name == 'UTNet':
        from model.trans_model.utnet import UTNet
        net = UTNet(channels, base_chan=32,num_classes=num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
   
    elif net_name =='TransUNet':
        from model.trans_model.transunet import VisionTransformer as ViT_seg
        from model.trans_model.transunet import CONFIGS as CONFIGS_ViT_seg
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = num_classes 
        config_vit.n_skip = 3 
        config_vit.patches.grid = (int(input_shape[0]/16), int(input_shape[1]/16))
        net = ViT_seg(config_vit, img_size=input_shape[0], num_classes=num_classes)
    
    return net


def eval_process(test_path,config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.device
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    # data loader
    test_transformer = transforms.Compose([
                Trunc_and_Normalize(config.scale),
                Get_ROI(pad_flag=False) if config.get_roi else transforms.Lambda(lambda x:x),
                CropResize(dim=config.input_shape,num_class=config.num_classes,crop=config.crop),
                To_Tensor(num_class=config.num_classes)
            ])

    test_dataset = DataGenerator(test_path,
                                roi_number=config.roi_number,
                                num_class=config.num_classes,
                                transform=test_transformer)

    test_loader = DataLoader(test_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    
    # get weight
    weight_path = get_weight_path(config.ckpt_path)
    print(weight_path)

    s_time = time.time()
    # get net
    net = get_net(config.net_name,
            config.encoder_name,
            config.channels,
            config.num_classes,
            config.input_shape,
            aux_deepvision=config.aux_deepvision,
            aux_classifier=config.aux_classifier
    )
    checkpoint = torch.load(weight_path,map_location='cpu')
    # print(checkpoint['state_dict'])
    msg=net.load_state_dict(checkpoint['state_dict'],strict=False)
    
    print(msg)
    get_net_time = time.time() - s_time
    print('define net and load weight need time:%.3f'%(get_net_time))

    pred = []
    true = []
    s_time = time.time()
    # net = net.cuda()
    # print(device)
    net = net.to(device)
    net.eval()
    move_time = time.time() - s_time
    print('move net to GPU need time:%.3f'%(move_time))

    extra_time = 0.
    with torch.no_grad():
        for step, sample in enumerate(test_loader):
            data = sample['image']
            target = sample['mask']
            ####
            # data = data.cuda()
            data = data.to(device)
            with autocast(True):
                output = net(data)
                
            if isinstance(output,tuple) or isinstance(output,list):
                seg_output = output[0]
            else:
                seg_output = output
            # seg_output = torch.argmax(torch.softmax(seg_output, dim=1),1).detach().cpu().numpy() 
            seg_output = torch.argmax(seg_output,1).detach().cpu().numpy()                           
            s_time = time.time()
            target = torch.argmax(target,1).detach().cpu().numpy()
            extra_time += time.time() - s_time
            if config.get_roi:
                bboxs = torch.stack(sample['bbox'],dim=0).cpu().numpy().T
                seg_output,target = resize_and_pad(seg_output,target,config.num_classes,config.input_shape,bboxs)
            pred.append(seg_output)
            true.append(target)
    pred = np.concatenate(pred,axis=0).squeeze().astype(np.uint8)
    true = np.concatenate(true,axis=0).squeeze().astype(np.uint8)
    # print(pred.shape)
    print('extra time:%.3f'%extra_time)
    return pred,true,extra_time+move_time+get_net_time


class Config:

    num_classes_dict = {
        'HaN_GTV':2,
        'THOR_GTV':2

    }
    scale_dict = {
        'HaN_GTV':[-150,200],
        'THOR_GTV':[-800,400]
    }

    roi_dict = {
        'HaN_GTV':'GTV',
        'THOR_GTV':'GTV'
    }
    
    input_shape = (512,512) #(512,512)(96,256,256)
    channels = 1
    crop = 0
    roi_number = 1
    batch_size = 32
    
    disease = 'HaN_GTV'
    mode = 'seg'
    num_classes = num_classes_dict[disease]
    scale = scale_dict[disease]

    two_stage = True
    
    net_name = 'sanet'
    encoder_name = 'resnet18'
    version = 'v7.1-roi-all'
    
    fold = 1
    device = "0"
    roi_name = roi_dict[disease]
    
    get_roi = False if 'roi' not in version else True
    aux_deepvision = False if 'sup' not in version else True
    aux_classifier = mode != 'seg'
    ckpt_path = f'./ckpt/{disease}/{mode}/{version}/{roi_name}'
    post_fix = '_quar'


if __name__ == '__main__':

    # test data
    data_path_dict = {
        'HaN_GTV':'../HaN_GTV/2d_test_data',
        # 'HaN_GTV':'../HaN_GTV/3d_test_data',
        'THOR_GTV':'../Thor_GTV/2d_test_data',
        # 'THOR_GTV':'../Thor_GTV/3d_test_data',
        
    }
    cls_result_dict = {
        # 'HaN_GTV': '../HaN_GTV/v1.0-roi-equal/GTV_vote.csv',
        # 'HaN_GTV': '../HaN_GTV/v1.0-roi-half/GTV_vote.csv', 
        'HaN_GTV': '../HaN_GTV/v1.0-roi-quar/GTV_vote.csv',
        # 'HaN_GTV': '../HaN_GTV/v1.0-roi-all/GTV_vote.csv',       
        # 'HaN_GTV': './result/HaN_GTV/mtl/v7.1-roi-all/GTV/vote.csv',
        
        'THOR_GTV': '../THOR_GTV/v1.0-roi-equal/GTV_vote.csv',
        # 'THOR_GTV': '../THOR_GTV/v1.0-roi-half/GTV_vote.csv',
        # 'THOR_GTV': '../THOR_GTV/v1.0-roi-quar/GTV_vote.csv',
        # 'THOR_GTV': '../THOR_GTV/v1.0-roi-all/GTV_vote.csv',   
        # 'THOR_GTV':'./result/THOR_GTV/mtl/v7.1-roi-all/GTV/vote.csv'
    }
    
    start = time.time()
    config = Config()
    data_path = data_path_dict[config.disease]
    sample_list = list(set([case.name.split('_')[0] for case in os.scandir(data_path)]))
    sample_list.sort()
    
    cls_result = csv_reader_single(cls_result_dict[config.disease],'id','pred') 

    ensemble_result = {}
    for fold in range(1,6):
        print('>>>>>>>>>>>> Fold%d >>>>>>>>>>>>'%fold)
        total_dice = []
        total_hd = []
        info_dice = []
        info_hd = []

        total_recall = []
        total_precision = []
        info_recall = []
        info_precision = []

        total_jc = []
        total_vs = []
        info_jc = []
        info_vs = []

        config.fold = fold
        config.ckpt_path = f'./ckpt/{config.disease}/{config.mode}/{config.version}/{config.roi_name}/fold{str(fold)}'
        save_dir = f'./result/{config.disease}/{config.mode}/{config.version}/{config.roi_name}{config.post_fix}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for sample in sample_list:
            info_item_dice = []
            info_item_hd = []
            info_item_dice.append(sample)
            info_item_hd.append(sample)

            info_item_recall = []
            info_item_precision = []
            info_item_recall.append(sample)
            info_item_precision.append(sample)

            info_item_jc = []
            info_item_vs = []
            info_item_jc.append(sample)
            info_item_vs.append(sample)

            print('>>>>>>>>>>>> %s is being processed'%sample)
            test_path = [case.path for case in os.scandir(data_path) if case.name.split('_')[0] == sample]
            data_len = len(test_path)
            print('data len: %d'%data_len)

            if len(config.input_shape) == 2:
                test_path.sort(key=lambda x:eval(x.split('_')[-1].split('.')[0]))
            
            # get end_index and start_index
            if config.two_stage:
                sample_index = [cls_result[ID] for ID in test_path]
                nonzero_index = np.nonzero(np.asarray(sample_index))[0]
                s_index, e_index = np.min(nonzero_index), np.max(nonzero_index)
            else:
                nonzero_index = np.asarray(range(data_len))
                s_index, e_index = 0, data_len
            ##
            sample_start = time.time()

            img = np.stack([hdf5_reader(item,'image') for item in test_path],axis=0)

            pred = np.zeros((data_len,) + config.input_shape, dtype=np.uint8)
            true = np.stack([hdf5_reader(item,'label').astype(np.uint8) for item in test_path],axis=0)
            assert true.shape[0] == data_len
            test_path = [test_path[index] for index in list(nonzero_index)]
            pred[nonzero_index],true[nonzero_index],extra_time = eval_process(test_path,config)
            
            pred = np.squeeze(pred)
            true = np.squeeze(true)

            total_time = time.time() - sample_start 
            actual_time = total_time - extra_time
            print('total time:%.3f'%total_time)
            print('actual time:%.3f'%actual_time)
            print("actual fps:%.3f"%(len(test_path)/actual_time))
            # print(pred.shape,true.shape)

            ############ dice & hd
            category_dice, avg_dice = multi_dice(true,pred,config.num_classes - 1)
            total_dice.append(category_dice)
            print('category dice:',category_dice)
            print('avg dice: %.3f'% avg_dice)
            # print(pred.shape,true.shape)

            category_hd, avg_hd = multi_hd(true,pred,config.num_classes - 1)
            total_hd.append(category_hd)
            print('category hd:',category_hd)
            print('avg hd: %.3f'% avg_hd)


            info_item_dice.extend(category_dice)
            info_item_hd.extend(category_hd)

            info_dice.append(info_item_dice)
            info_hd.append(info_item_hd)

            ############

            ############ recall & precision for binary output
            recall = recall_score(true.flatten(),pred.flatten())
            precision = precision_score(true.flatten(),pred.flatten())
            
            total_recall.append(recall)
            print('category recall:%.3f'%recall)

            total_precision.append(precision)
            print('category precision:%.3f'%precision)

            info_item_recall.append(recall)
            info_item_precision.append(precision)

            info_recall.append(info_item_recall)
            info_precision.append(info_item_precision)

            ############


            ############ jc and vs for binary output
            category_jc, avg_jc = multi_jc(true,pred,config.num_classes - 1)
            category_vs, avg_vs = multi_vs(true,pred,config.num_classes - 1)
            
            total_jc.append(category_jc)
            print('category jc:',category_jc)

            total_vs.append(category_vs)
            print('category vs:',category_vs)

            info_item_jc.extend(category_jc)
            info_item_vs.extend(category_vs)

            info_jc.append(info_item_jc)
            info_vs.append(info_item_vs)

            ############

            if sample not in ensemble_result:
                ensemble_result[sample] = {
                    'true':[true],
                    'pred':[]
                }
            ensemble_result[sample]['pred'].append(pred)

        dice_csv = pd.DataFrame(data=info_dice)
        hd_csv = pd.DataFrame(data=info_hd)

        recall_csv = pd.DataFrame(data=info_recall)
        precision_csv = pd.DataFrame(data=info_precision)

        jc_csv = pd.DataFrame(data=info_jc)
        vs_csv = pd.DataFrame(data=info_vs)

        if not config.two_stage:
            dice_csv.to_csv(os.path.join(save_dir,f'fold{config.fold}_dice.csv'))
            hd_csv.to_csv(os.path.join(save_dir,f'fold{config.fold}_hd.csv'))

            recall_csv.to_csv(os.path.join(save_dir,f'fold{config.fold}_recall.csv'))
            precision_csv.to_csv(os.path.join(save_dir,f'fold{config.fold}_precision.csv'))

            jc_csv.to_csv(os.path.join(save_dir,f'fold{config.fold}_jc.csv'))
            vs_csv.to_csv(os.path.join(save_dir,f'fold{config.fold}_vs.csv'))
        else:
            dice_csv.to_csv(os.path.join(save_dir,f'ts_fold{config.fold}_dice.csv'))
            hd_csv.to_csv(os.path.join(save_dir,f'ts_fold{config.fold}_hd.csv'))

            recall_csv.to_csv(os.path.join(save_dir,f'ts_fold{config.fold}_recall.csv'))
            precision_csv.to_csv(os.path.join(save_dir,f'ts_fold{config.fold}_precision.csv'))

            jc_csv.to_csv(os.path.join(save_dir,f'ts_fold{config.fold}_jc.csv'))
            vs_csv.to_csv(os.path.join(save_dir,f'ts_fold{config.fold}_vs.csv'))

        total_dice = np.stack(total_dice,axis=0) #sample*classes
        total_category_dice = np.mean(total_dice,axis=0)
        total_avg_dice = np.mean(total_category_dice)

        print('total category dice mean:',total_category_dice)
        print('total category dice std:',np.std(total_dice,axis=0))
        print('total dice mean: %.3f'% total_avg_dice)

        total_hd = np.stack(total_hd,axis=0) #sample*classes
        total_category_hd = np.mean(total_hd,axis=0)
        total_avg_hd = np.mean(total_category_hd)

        print('total category hd mean:',total_category_hd)
        print('total category hd std:',np.std(total_hd,axis=0))
        print('total hd mean: %.3f'% total_avg_hd)

        ##### for binary output
        print('total recall mean:',np.mean(total_recall))
        print('total recall std:',np.std(total_recall))
        print('total precision mean:', np.mean(total_precision))
        print('total precision std:',np.std(total_precision))


        total_jc = np.stack(total_jc,axis=0) #sample*classes
        total_category_jc = np.mean(total_jc,axis=0)
        total_avg_jc = np.mean(total_category_jc)

        print('total category jc mean:',total_category_jc)
        print('total category jc std:',np.std(total_jc,axis=0))
        print('total jc mean: %.3f'% total_avg_jc)

        total_vs = np.stack(total_vs,axis=0) #sample*classes
        total_category_vs = np.mean(total_vs,axis=0)
        total_avg_vs = np.mean(total_category_vs)

        print('total category vs mean:',total_category_vs)
        print('total category vs std:',np.std(total_vs,axis=0))
        print('total vs mean: %.3f'% total_avg_vs)
        #####

        print("runtime:%.3f"%(time.time() - start))

    #### for ensemble and post-processing

    ensemble_info_dice = []
    ensemble_info_hd = []
    post_ensemble_info_dice = []
    post_ensemble_info_hd = []


    ensemble_info_recall = []
    ensemble_info_precision = []
    post_ensemble_info_recall = []
    post_ensemble_info_precision = []


    ensemble_info_jc = []
    ensemble_info_vs = []
    post_ensemble_info_jc = []
    post_ensemble_info_vs = []

    for sample in sample_list:
        print('>>>> %s in post processing'%sample)
        ensemble_pred = ensemble(np.stack(ensemble_result[sample]['pred'],axis=0),config.num_classes - 1)
        ensemble_true = ensemble_result[sample]['true'][0]

        
        category_dice, avg_dice = multi_dice(ensemble_true,ensemble_pred,config.num_classes - 1)
        category_hd, avg_hd = multi_hd(ensemble_true,ensemble_pred,config.num_classes - 1)

        ensemble_recall = recall_score(ensemble_true.flatten(),ensemble_pred.flatten())
        ensemble_precision = precision_score(ensemble_true.flatten(),ensemble_pred.flatten())

        category_jc, avg_jc = multi_jc(ensemble_true,ensemble_pred,config.num_classes - 1)
        category_vs, avg_vs = multi_vs(ensemble_true,ensemble_pred,config.num_classes - 1)


        post_ensemble_pred = post_seg(ensemble_pred,list(range(1,config.num_classes)),keep_max=config.disease=='HaN_GTV')
        post_category_dice, post_avg_dice = multi_dice(ensemble_true,post_ensemble_pred,config.num_classes - 1)
        post_category_hd, post_avg_hd = multi_hd(ensemble_true,post_ensemble_pred,config.num_classes - 1)

        # print(np.unique(post_ensemble_pred))
        post_ensemble_recall = recall_score(ensemble_true.flatten(),post_ensemble_pred.flatten())
        post_ensemble_precision = precision_score(ensemble_true.flatten(),post_ensemble_pred.flatten())

        post_category_jc, post_avg_jc = multi_jc(ensemble_true,post_ensemble_pred,config.num_classes - 1)
        post_category_vs, post_avg_vs = multi_vs(ensemble_true,post_ensemble_pred,config.num_classes - 1)

        ### save result as nii
        # from utils import save_as_nii
        # nii_path = os.path.join(save_dir,'nii')
        # if not os.path.exists(nii_path):
        #     os.makedirs(nii_path)
        # img_path = os.path.join(nii_path, sample + '_image.nii.gz')
        # lab_path = os.path.join(nii_path, sample + '_label.nii.gz')
        # save_as_nii(img,img_path)
        # save_as_nii(post_ensemble_pred,lab_path) 
        ###

        print('ensemble recall:', ensemble_recall)
        print('ensemble precision:', ensemble_precision)

        print('post ensemble recall:', post_ensemble_recall)
        print('post ensemble precision:', post_ensemble_precision)


        print('ensemble category dice:',category_dice)
        print('ensemble avg dice: %.3f'% avg_dice)
        print('ensemble category hd:',category_hd)
        print('ensemble avg hd: %.3f'% avg_hd)


        print('ensemble category jc:',category_jc)
        print('ensemble avg jc: %.3f'% avg_jc)
        print('ensemble category vs:',category_vs)
        print('ensemble avg vs: %.3f'% avg_vs)

        print('post ensemble category dice:',post_category_dice)
        print('post ensemble avg dice: %.3f'% post_avg_dice)
        print('post ensemble category hd:',post_category_hd)
        print('post ensemble avg hd: %.3f'% post_avg_hd)

        print('post ensemble category jc:',post_category_jc)
        print('post ensemble avg jc: %.3f'% post_avg_jc)
        print('post ensemble category vs:',post_category_vs)
        print('post ensemble avg vs: %.3f'% post_avg_vs)
        

        ensemble_item_dice = [sample]
        ensemble_item_hd = [sample]
        post_ensemble_item_dice = [sample]
        post_ensemble_item_hd = [sample]

        ensemble_item_recall = [sample]
        ensemble_item_precision = [sample]
        post_ensemble_item_recall = [sample]
        post_ensemble_item_precision = [sample]

        ensemble_item_jc = [sample]
        ensemble_item_vs = [sample]
        post_ensemble_item_jc = [sample]
        post_ensemble_item_vs = [sample]

        
        ensemble_item_dice.extend(category_dice)
        ensemble_item_hd.extend(category_hd)
        post_ensemble_item_dice.extend(post_category_dice)
        post_ensemble_item_hd.extend(post_category_hd)


        ensemble_item_recall.append(ensemble_recall)
        ensemble_item_precision.append(ensemble_precision)
        post_ensemble_item_recall.append(post_ensemble_recall)
        post_ensemble_item_precision.append(post_ensemble_precision)
        

        ensemble_item_jc.extend(category_jc)
        ensemble_item_vs.extend(category_vs)
        post_ensemble_item_jc.extend(post_category_jc)
        post_ensemble_item_vs.extend(post_category_vs)


        ensemble_info_dice.append(ensemble_item_dice)
        ensemble_info_hd.append(ensemble_item_hd)
        post_ensemble_info_dice.append(post_ensemble_item_dice)
        post_ensemble_info_hd.append(post_ensemble_item_hd)


        ensemble_info_recall.append(ensemble_item_recall)
        ensemble_info_precision.append(ensemble_item_precision)
        post_ensemble_info_recall.append(post_ensemble_item_recall)
        post_ensemble_info_precision.append(post_ensemble_item_precision)

        ensemble_info_jc.append(ensemble_item_jc)
        ensemble_info_vs.append(ensemble_item_vs)
        post_ensemble_info_jc.append(post_ensemble_item_jc)
        post_ensemble_info_vs.append(post_ensemble_item_vs)
    
    

    ensemble_dice_csv = pd.DataFrame(data=ensemble_info_dice)
    ensemble_hd_csv = pd.DataFrame(data=ensemble_info_hd)
    post_ensemble_dice_csv = pd.DataFrame(data=post_ensemble_info_dice)
    post_ensemble_hd_csv = pd.DataFrame(data=post_ensemble_info_hd)


    ensemble_recall_csv = pd.DataFrame(data=ensemble_info_recall)
    ensemble_precision_csv = pd.DataFrame(data=ensemble_info_precision)
    post_ensemble_recall_csv = pd.DataFrame(data=post_ensemble_info_recall)
    post_ensemble_precision_csv = pd.DataFrame(data=post_ensemble_info_precision)


    ensemble_jc_csv = pd.DataFrame(data=ensemble_info_jc)
    ensemble_vs_csv = pd.DataFrame(data=ensemble_info_vs)
    post_ensemble_jc_csv = pd.DataFrame(data=post_ensemble_info_jc)
    post_ensemble_vs_csv = pd.DataFrame(data=post_ensemble_info_vs)

    if not config.two_stage:
        ensemble_dice_csv.to_csv(os.path.join(save_dir,'ensemble_dice.csv'))
        ensemble_hd_csv.to_csv(os.path.join(save_dir,'ensemble_hd.csv'))
        post_ensemble_dice_csv.to_csv(os.path.join(save_dir,'post_ensemble_dice.csv'))
        post_ensemble_hd_csv.to_csv(os.path.join(save_dir,'post_ensemble_hd.csv'))

        ensemble_recall_csv.to_csv(os.path.join(save_dir,'ensemble_recall.csv'))
        ensemble_precision_csv.to_csv(os.path.join(save_dir,'ensemble_precision.csv'))
        post_ensemble_recall_csv.to_csv(os.path.join(save_dir,'post_ensemble_recall.csv'))
        post_ensemble_precision_csv.to_csv(os.path.join(save_dir,'post_ensemble_precision.csv'))

        ensemble_jc_csv.to_csv(os.path.join(save_dir,'ensemble_jc.csv'))
        ensemble_vs_csv.to_csv(os.path.join(save_dir,'ensemble_vs.csv'))
        post_ensemble_jc_csv.to_csv(os.path.join(save_dir,'post_ensemble_jc.csv'))
        post_ensemble_vs_csv.to_csv(os.path.join(save_dir,'post_ensemble_vs.csv'))

    else:
        ensemble_dice_csv.to_csv(os.path.join(save_dir,'ts_ensemble_dice.csv'))
        ensemble_hd_csv.to_csv(os.path.join(save_dir,'ts_ensemble_hd.csv'))
        post_ensemble_dice_csv.to_csv(os.path.join(save_dir,'ts_post_ensemble_dice.csv'))
        post_ensemble_hd_csv.to_csv(os.path.join(save_dir,'ts_post_ensemble_hd.csv'))

        ensemble_recall_csv.to_csv(os.path.join(save_dir,'ts_ensemble_recall.csv'))
        ensemble_precision_csv.to_csv(os.path.join(save_dir,'ts_ensemble_precision.csv'))
        post_ensemble_recall_csv.to_csv(os.path.join(save_dir,'ts_post_ensemble_recall.csv'))
        post_ensemble_precision_csv.to_csv(os.path.join(save_dir,'ts_post_ensemble_precision.csv'))

        ensemble_jc_csv.to_csv(os.path.join(save_dir,'ts_ensemble_jc.csv'))
        ensemble_vs_csv.to_csv(os.path.join(save_dir,'ts_ensemble_vs.csv'))
        post_ensemble_jc_csv.to_csv(os.path.join(save_dir,'ts_post_ensemble_jc.csv'))
        post_ensemble_vs_csv.to_csv(os.path.join(save_dir,'ts_post_ensemble_vs.csv'))

    #### end