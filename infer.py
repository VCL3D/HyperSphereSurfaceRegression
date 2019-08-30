import os
import datetime
import argparse

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

import file_utils as fu
import nn_utils as nu
import numpy as np
import cv2

'''
    Global argument parser
'''
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--input", type = str, help = "Absolute path to the input RGB image")
arg_parser.add_argument("--output", type = str, help = "Desired filepath to save the model's prediction")
arg_parser.add_argument("--weights", type = str, help = "Absolute path to the trained model weights")
arg_parser.add_argument("--png", action = "store_true", help = "Set this flag to save the output nsurface map as .png too")
# arguments
args = arg_parser.parse_args()
cli_args = vars(args)

'''
    Simple logger
'''
class Logger:
    def __init__(self, log_filepath):
        self.log_filepath = log_filepath
        if self.log_filepath is not None:
            log_file = open(self.log_filepath, 'w')
            log_file.close()


    def log(self, message):
        line = "{} | {}".format(datetime.datetime.now(), message)
        print(line)
        if self.log_filepath is not None:
            log_file = open(self.log_filepath, 'a')
            line += "\n"
            log_file.write(line)
            log_file.close()

'''
    Global Logger object
'''
logger = Logger(None)

def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR)
    if img.shape[1] != 512 or img.shape[0] != 256:
        img = cv2.resize(img, (256, 512), cv2.INTER_LINEAR)
    return img

def numpy_to_tensor(np_array):
    np_array = np_array.transpose(2, 0, 1)
    tensor = torch.from_numpy(np_array)
    return torch.as_tensor(tensor, dtype = torch.float32).unsqueeze(0)


def main(args):
    logger.log("Infering:")
    logger.log("\tInput img       : {}".format(args['input']))
    logger.log("\tSaving output to: {}".format(args['output']))

    logger.log("Configuring device...")
    device = nu.configure_device(None)

    logger.log("Loading model...")
    model = nu.load_model(True).to(device)
    model = nu.load_model_and_optimizer_state(model, None, args['weights'])

    logger.log("Infering...")
    with torch.no_grad():
        input_rgb = numpy_to_tensor(load_image(args['input'])).to(device)
        pred = model(input_rgb)
        pred = F.normalize(pred, p = 2, dim = 1)

        logger.log("Saving prediction...")
        directory = os.path.dirname(args['output'])
        filename = os.path.basename(args['output'])

        fu.save_tensor_as_float(directory, filename, pred, args['png'])
    logger.log("Done.")
    

if __name__ == "__main__":
    main(cli_args)