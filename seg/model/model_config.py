MODEL_CONFIG = {
    'unet':{
        'simplenet':{
            'in_channels':1,
            'encoder_name':'simplenet',
            'encoder_depth':5,
            'encoder_channels':[32,64,128,256,512],  #[1,2,4,8,16]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[8,4,2,1]
            'upsampling':1,
            'classes':2,
            'aux_classifier': False
        },
        'resnet18':{
            'in_channels':1,
            'encoder_name':'resnet18',
            'encoder_depth':5,
            'encoder_channels':[64,64,128,256,512],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'aux_classifier': False
        },
        'swin_transformer':{
            'in_channels':1,
            'encoder_name':'swin_transformer',
            'encoder_depth':4,
            'encoder_channels':[96,192,384,768],  #[4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64], #[16,8,4]
            'upsampling':4,
            'classes':2,
            'aux_classifier': False
        },
        'resnet50':{
            'in_channels':1,
            'encoder_name':'resnet50',
            'encoder_depth':5,
            'encoder_channels':[64,256,512,1024,2048],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'aux_classifier': False
        }
    },
    # att unet
    'att_unet':{
        'simplenet':{
            'in_channels':1,
            'encoder_name':'simplenet',
            'encoder_depth':5,
            'encoder_channels':[32,64,128,256,512],  #[1,2,4,8,16]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[8,4,2,1]
            'upsampling':1,
            'classes':2,
            'aux_classifier': False
        },
        'swin_transformer':{
            'in_channels':1,
            'encoder_name':'swin_transformer',
            'encoder_depth':4,
            'encoder_channels':[96,192,384,768],  #[4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64], #[16,8,4]
            'upsampling':4,
            'classes':2,
            'aux_classifier': False
        },
        'resnet18':{
            'in_channels':1,
            'encoder_name':'resnet18',
            'encoder_depth':5,
            'encoder_channels':[64,64,128,256,512],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'aux_classifier': False
        },
        'resnet50':{
            'in_channels':1,
            'encoder_name':'resnet50',
            'encoder_depth':5,
            'encoder_channels':[64,256,512,1024,2048],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'aux_classifier': False
        }
    },
    # res unet
    'res_unet':{
        'simplenet':{
            'in_channels':1,
            'encoder_name':'simplenet_res',
            'encoder_depth':5,
            'encoder_channels':[32,64,128,256,512],  #[1,2,4,8,16]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[8,4,2,1]
            'upsampling':1,
            'classes':2,
            'use_center':False,
            'aux_classifier': False,
            'aux_deepvision':False
        },
        'resnet18':{
            'in_channels':1,
            'encoder_name':'resnet18',
            'encoder_depth':5,
            'encoder_channels':[64,64,128,256,512],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'use_center':False,
            'aux_classifier': False,
            'aux_deepvision':False
        },
        'resnet50':{
            'in_channels':1,
            'encoder_name':'resnet50',
            'encoder_depth':5,
            'encoder_channels':[64,256,512,1024,2048],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'use_center':False,
            'aux_classifier': False,
            'aux_deepvision':False
        }
    },
    # sanet
    'sanet':{
        'simplenet':{
            'in_channels':1,
            'encoder_name':'simplenet',
            'encoder_weights':None,
            'encoder_depth':5,
            'encoder_channels':[32,64,128,256,512],  #[1,2,4,8,16][-num_stage:]
            'num_stage':4,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[128], 
            'upsampling':2,
            'classes':2,
            'aux_classifier': False
        },
        'resnet18':{
            'in_channels':1,
            'encoder_name':'resnet18',
            'encoder_weights':None,
            'encoder_depth':5,
            'encoder_channels':[64,64,128,256,512],  #[2,4,8,16,32]
            'encoder_output_stride':16,
            'num_stage':4,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[128], #[16,8,4,2]
            'upsampling':4,
            'classes':2,
            'aux_classifier': False
        },
        'resnet50':{
            'in_channels':1,
            'encoder_name':'resnet50',
            'encoder_weights':None,
            'encoder_depth':5,
            'encoder_channels':[64,256,512,1024,2048],  #[2,4,8,16,32]
            'encoder_output_stride':16,
            'num_stage':4,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[128], #[16,8,4,2]
            'upsampling':4,
            'classes':2,
            'aux_classifier': False
        },
        'swin_transformer':{
            'in_channels': 1,
            'encoder_name': 'swin_transformer',
            'encoder_weights': None,
            'encoder_depth': 4,
            'encoder_channels':[96,192,384,768],  #[4,8,16,32]
            'num_stage':4,
            'decoder_use_batchnorm': True,
            'decoder_attention_type':'cbam',
            'decoder_channels': [256],
            'upsampling':4,
            'classes':1,
            'aux_classifier':False
        }
    },
}