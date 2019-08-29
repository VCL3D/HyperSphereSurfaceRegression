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

'''
    main function
'''
def main(cli_args):
    settings = fu.read_configuration_file(cli_args['conf'])
    train(settings)


'''
    Trains model
'''
def train(settings):
    logger.log("Initializing training...")
    
    logger.log("Configuring Device...")
    gpus = settings['session']['gpu']
    device = nu.configure_device(gpus)

    logger.log("Configuring Model...")
    model = None
    if settings['session']['pretrained']:
        logger.log("Loading pre-trained weights...")
        model = nu.load_"model(True)
    else:
        model = nu.load_model(False)
    logger.log("Initializing model weights (using Xavier initialization)...")
    model.init_weights()
    model.to(device)

    logger.log("Configuring Optimizer...")
    optim_settings = settings["session"]["optimizer"]
    opt = optim_settings['optim']
    lr = optim_settings['lr']
    wd = optim_settings['weight_decay']
    mom = optim_settings['momentum']
    mom2 = optim_settings['momentum2']
    eps = optim_settings['epsilon']
    optim_params = nu.OptimParams(lr, mom, mom2, eps, wd)
    optimizer = nu.get_optimizer(opt, model.parameters(), optim_params)
    # optimizer.to(device)

    logger.log("Preseeding...")
    nu.preseed(settings['session']['seed'])

    # make train loader
    logger.log("Configuring data loader...")
    train_bsize = settings['session']['train_batch_size']
    eval_bsize = settings['session']['eval_batch_size']
    train_set = dataset360N.Dataset360N(
        settings["session"]["train_filenames_filepath"],
        " ",
        settings["session"]["input_shape"])
    eval_set = dataset360N.Dataset360N(
        settings["session"]["validation_filenames_filepath"],
        " ",
        settings["session"]["input_shape"])
    train_loader = DataLoader(train_set, batch_size = train_bsize, shuffle = True, pin_memory = True)
    eval_loader = DataLoader(eval_set, batch_size = eval_bsize, shuffle = True, pin_memory = True)

    epochs = settings['session']["epochs"]
    epoch_range = range(epochs)
    disp_iters = settings['session']["display_iterations"]
    chkp_iters = settings['session']["chkp_iterations"]
    eval_iters = settings['session']["evaluation_iterations"]
    chkp_path = settings['session']["chkp_path"]
    sess_name = settings['session']["session_name"]
    alpha = settings['session']["loss"]["alpha"]
    
    logger.log("Training...")
    g_iters = 0
    for e in epoch_range:
        for b_idx, train_sample in enumerate(train_loader):
            active_loss = torch.tensor(0.0).to(device)
            quaternion_loss = 0.0
            smoothness_loss = 0.0

            rgb = train_sample["input_rgb"].to(device)
            target = train_sample["target_surface"].to(device)
            mask = train_sample["mask"].to(device)

            pred = model(rgb)
            pred = F.normalize(pred, p = 2, dim = 1)

            quat_loss, quat_loss_map = losses.quaternion_loss(pred, target, True, mask)
            quaternion_loss += quat_loss * (1 - alpha)
            smooth_loss, smooth_loss_map = losses.smoothness_loss(pred, True, mask)
            smoothness_loss += smooth_loss * alpha

            active_loss = quat_loss * (1 - alpha) + smooth_loss * (alpha)

            optimizer.zero_grad()
            active_loss.backward()
            optimizer.step()

            g_iters += train_bsize
            if g_iters % chkp_iters == 0:
                logger.log("Saving Checkpoint in: {}".format(chkp_path))
                fu.save_state(chkp_path, sess_name, model, optimizer, e + 1, g_iters)
            if g_iters % disp_iters == 0:
                logger.log("Epoch: {} | Training iter: {} | Training Loss:".format(e + 1, g_iters))
                logger.log("\t\t\t\t\tTotal Loss     : {}".format(active_loss))
                logger.log("\t\t\t\t\tQuaternion Loss: {}".format(quat_loss))
                logger.log("\t\t\t\t\tSmoothness Loss: {}".format(smooth_loss))
            if g_iters % eval_iters == 0:
                logger.log("Evaluating...")
                model.eval()
                eval_loss = 0.0
                counter = 0.0
                with torch.no_grad():
                    active_loss = torch.tensor(0.0).to(device)
                    for eval_b_idx, eval_sample in enumerate(eval_loader):
                        quaternion_loss = 0.0
                        smoothness_loss = 0.0

                        rgb = train_sample["input_rgb"].to(device)
                        target = train_sample["target_surface"].to(device)
                        mask = train_sample["mask"].to(device)

                        pred = model(rgb)
                        pred = F.normalize(pred, p = 2, dim = 1)

                        quat_loss, quat_loss_map = losses.quaternion_loss(pred, target, True, mask)
                        quaternion_loss += quat_loss * (1 - alpha)
                        smooth_loss, smooth_loss_map = losses.smoothness_loss(pred, True, mask)
                        smoothness_loss += smooth_loss * alpha

                        active_loss += quat_loss * (1 - alpha) + smooth_loss * (alpha)
                        counter += eval_bsize
                    total_loss = active_loss / counter
                    logger.log("Evaluation finished. Total Loss: {}".format(total_loss))
        logger.log("Epoch {} finished.".format(e + 1))
    logger.log("Training session finished.")


if __name__ == "__main__":
    main(cli_args)