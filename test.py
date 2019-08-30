import os
import datetime
import argparse

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import file_utils as fu
import nn_utils as nu
import losses
import dataset360N

'''
    Global argument parser
'''
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--conf", type = str, help = "Absolute path to the configuration file")
arg_parser.add_argument("--log", type = str, help = "Directory to save log file")
# arguments
args = arg_parser.parse_args()
cli_args = vars(args)

'''
    Simple logger
'''
class Logger:
    def __init__(self, log_filepath):
        self.log_filepath = log_filepath
        log_file = open(self.log_filepath, 'w')
        log_file.close()


    def log(self, message):
        line = "{} | {}".format(datetime.datetime.now(), message)
        print(line)
        log_file = open(self.log_filepath, 'a')
        line += "\n"
        log_file.write(line)
        log_file.close()

'''
    Global Logger object
'''
logger = Logger(cli_args['log'])

def main(cli_args):
    settings = fu.read_configuration_file(cli_args['conf'])
    test(settings)

def test(settings):
    logger.log("Initializing testing...")
    logger.log("Configuring Device...")
    gpus = settings['session']['gpu']
    device = nu.configure_device(gpus)

    logger.log("Configuring model...")
    model = nu.load_model(True)
    if settings['session']['chkp_path']:
        model = nu.load_model_and_optimizer_state(model, None, settings['session']['chkp_path'])
    else:
        logger.log("Failed to load pre-trained weights. No valid checkpoint path <{}> was given".format(settings['session']['chkp_path']))
        exit()
    model.to(device)

    logger.log("Configuring data loader...")
    test_bsize = settings['session']['test_batch_size']
    test_set = dataset360N.Dataset360N(
        settings['session']['test_filenames_filepath'],
        " ",
        settings['session']['input_shape']
    )
    test_loader = DataLoader(test_set, batch_size = test_bsize, shuffle = False, pin_memory = True)

    logger.log("Testing...")
    total_loss = torch.tensor(0.0).to(device)
    with torch.no_grad():
        for b_idx, test_sample in enumerate(test_loader):
            active_loss = torch.tensor(0.0).to(device)

            rgb = test_sample['input_rgb'].to(device)
            target = test_sample['target_surface'].to(device)
            mask = test_sample['mask'].to(device)

            pred = model(rgb)
            pred = F.normalize(pred, p = 2, dim = 1)
            logger.log("Tested: {}".format(test_sample['filename']))
    logger.log("Testing finished.")

if __name__ == "__main__":
    main(cli_args)