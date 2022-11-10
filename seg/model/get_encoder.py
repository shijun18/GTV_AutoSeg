import sys
sys.path.append('..')
import torch
from model.encoder import swin_transformer,simplenet,resnet,mobilenetv3,xception,efficientnet


ssl_weight_path = {
    'resnet18':None,
    'resnet50':None,
}

def build_encoder(arch='resnet18', weights=None, **kwargs):
        
    arch = arch.lower()
    
    if arch.startswith('resnet'):
        backbone = resnet.__dict__[arch](classification=False,pretrained=False or weights=='imagenet',**kwargs)
    elif arch.startswith('swin_transformer'):
        backbone = swin_transformer.__dict__[arch](classification=False,**kwargs)
    elif arch.startswith('simplenet'):
        backbone = simplenet.__dict__[arch](**kwargs)
    elif arch.startswith('mobilenetv3'):
        backbone = mobilenetv3.__dict__[arch](**kwargs)
    elif arch.startswith('xception'):
        backbone = xception.__dict__[arch](**kwargs)
    elif arch.startswith('timm_efficientnet'):
        backbone = efficientnet.__dict__[arch](pretrained=False or weights=='imagenet',**kwargs)
    
    else:
        raise Exception('Architecture undefined!')

    if weights == 'ssl' and isinstance(ssl_weight_path[arch], str):
        print('Loading weights for backbone')
        msg = backbone.load_state_dict(
            torch.load(ssl_weight_path[arch], map_location=lambda storage, loc: storage)['state_dict'], strict=False)
        # if arch.startswith('resnet'):
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print(">>>> loaded pre-trained model '{}' ".format(ssl_weight_path[arch]))
        print('>>>> missing keys:',msg[0])
    
    return backbone

