import os
import json
import torch
import numpy as np
import cv2

'''
    Reads configuration file
    \param
        filepath        the absolute path to the configuration file
    \return 
        settings_map    dictionary with the configuration settings
'''
def read_configuration_file(filepath):
    print("Reading configuration file...")
    settings = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as fd:
            settings = json.load(fd)
            assert settings['session'], print("Failed to read configuration file. No session settings.")
            # assert settings['session']['optimizer'], print("Failed to read configuration file. No optimizer settings.")
    return settings

def save_state(directory, session_name, model, optimizer, epoch, global_iters):
    if os.path.isfile(directory):
        directory = os.path.abspath(os.path.dirname(directory))
    model_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()
    model_filename = session_name + "_model_e_{}_b_{}.chkp".format(epoch, global_iters)
    optim_filename = session_name + "_optim_e_{}_b_{}.chkp".format(epoch, global_iters)
    model_filepath = os.path.join(directory, model_filename)
    optim_filepath = os.path.join(directory, optim_filename)
    torch.save(model_state_dict, model_filepath)
    torch.save(optim_state_dict, optim_filepath)
    
def save_tensor_as_float(directory, filename, tensor, png = False):
    if not os.path.exists(directory):
        print("Given directory does not exist. Creating...")
        os.mkdir(directory)
    tensor = tensor.detach().cpu()
    array = tensor.squeeze(0).numpy()
    array = array.transpose(1, 2, 0)
    filepath_exr = os.path.join(directory, filename + ".exr")
    cv2.imwrite(filepath_exr, array)
    if (png):
        norm_array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
        norm_array = norm_array.astype(np.uint8)
        filepath_png = os.path.join(directory, filename.replace("exr", "png"))
        cv2.imwrite(filepath_png, norm_array)