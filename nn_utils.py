import torch
import torch.optim as optim
import torchvision.models
import random
import numpy as np
import os
import model as M

class OptimParams(object):
    def __init__(self, lr = 0.0001, momentum = 0.9, momentum2 = 0.999, eps = 1e-8, weight_decay = 0.0, damp = 0):
        self.lr = lr
        self.momentum = momentum
        self.momentum2 = momentum2
        self.eps = eps
        self.damp = damp
        self.weight_decay = weight_decay
    
    def get_learning_rate(self):
        return self.lr
    
    def get_momentum(self):
        return self.momentum
    
    def get_momentum2(self):
        return self.momentum2
    
    def get_epsilon(self):
        return self.eps
    
    def get_weight_decay(self):
        return self.weight_decay
    
    def get_damp(self):
        return self.damp


def get_optimizer(optim_type, model_params, optim_params):
    if (optim_type == "adam"):
        return optim.Adam(
            model_params,
            lr = optim_params.get_learning_rate(),
            betas = (optim_params.get_momentum(), optim_params.get_momentum2()),
            eps = optim_params.get_epsilon(),
            weight_decay = optim_params.get_epsilon())
    else:
        print("Error: Given optimizer type <{}>, is not valid".format(optim_type))

# def init_optimizer(optim_type, model, optim_params, optim_state = None):
#     optimizer = get_optimizer(optim_type, model.parameters(), optim_params)
#     if optim_state is not None:
#         state = torch.load(optim_state)
#         print("Loading previously saved optimizer state from {}".format(optim_state))
#         optimizer.load_state_dict(state)
#     return optimizer

def configure_device(gpus):
    if (gpus is not None and torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0):
        device = torch.device('cuda:{}'.format(gpus[0]))
    elif (gpus is None and torch.cuda.is_available()):
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print("Selected Device: {}".format(device))
    return device


def preseed(seed):
    print("Preseeding for reproducibility with user seed: {}".format(seed))
    if seed > 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.benchmark = False
        random.seed(seed)

def load_model_and_optimizer_state(model, optimizer, chkp_path):
    if os.path.exists(chkp_path) and optimizer is None:
        print("Loading model checkpoint from: {}...".format(chkp_path))
        model_chkp = chkp_path
        model_state = torch.load(model_chkp)
        model.load_state_dict(model_state)
        return model
    if (os.path.exists(chkp_path)):
        model_chkp = chkp_path
        optim_chkp = chkp_path.replace("model", "optim")
        print("Loading model checkpoint from: {}...".format(model_chkp))
        print("Loading optimizer checkpoint from: {}...".format(optim_chkp))
        model_state = torch.load(model_chkp)
        optim_state = torch.load(optim_chkp)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)
        return model, optim

def load_model(trained = True):
    features = None
    if trained:
        orig_vgg = torchvision.models.vgg16(pretrained = trained)
        features = orig_vgg.features
    return M.VGG16Unet(features)

    