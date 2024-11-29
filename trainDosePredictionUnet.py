import numpy as np
import os
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Custom
from models import AttentionGenerator
from dataset import VolumesFromList
from config import load_config

import matplotlib.pyplot as plt
from util import *


def train(data_dir, train_list_path, test_list_path, save_dir, exp_name_base, exp_name, params):
    num_epochs = params["num_epochs"]
    log_interval = params["log_interval"]
    loss_type = params["loss_type"]
    batch_size = params["batch_size"]
    g_lr = params["g_lr"]

    model = AttentionGenerator(3, 1)
    model.cuda()

    opt = optim.Adam(model.parameters(), lr=g_lr, betas=(0.5, 0.999), )

    if loss_type == "l1":
        voxel_criterion = nn.L1Loss().cuda()
    elif loss_type == "l2":
        voxel_criterion = nn.MSELoss().cuda()

    train_dataset = VolumesFromList(data_dir, train_list_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    test_dataset = VolumesFromList(data_dir, test_list_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()

    log_path = os.path.join(save_dir, "Logs", exp_name_base, exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    writer = SummaryWriter(log_path)

    epoch_loop = tqdm.tqdm(range(num_epochs + 1))
    # for epoch in range(num_epochs):
    for epoch in epoch_loop:
        model.train()
        cum_loss = 0
        loss_counter = 0
        for idx, volumes in enumerate(train_loader):
            real_dose = volumes[:, 0, :, :, :].unsqueeze(1).float()
            ct = volumes[:, 1, :, :, :].unsqueeze(1).float()
            prescription = volumes[:, 2, :, :, :].unsqueeze(1).float()
            oars = volumes[:, 3, :, :, :].unsqueeze(1).float()

            real_dose = real_dose.cuda()
            ct = ct.cuda()
            prescription = prescription.cuda()
            oars = oars.cuda()

            model_input = torch.cat((ct, prescription, oars), dim=1)

            y_fake = model(model_input)

            loss = voxel_criterion(y_fake, real_dose)

            opt.zero_grad()
            g_scaler.scale(loss).backward()
            g_scaler.step(opt)
            g_scaler.update()

            cum_loss += loss.item()
            loss_counter += 1

        cum_loss /= loss_counter

        # save weights
        weights_path = os.path.join(save_dir, "Weights", exp_name_base, exp_name)
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)

        if epoch % 100 == 0 or epoch == num_epochs:
            torch.save(model.state_dict(),
                       os.path.join(weights_path, "Unet" + str(epoch) + ".pth"))

        # Testing
        if epoch % log_interval == 0:
            model.eval()
            cum_loss_test = 0
            test_loss_counter = 0
            for test_idx, test_volumes in enumerate(test_loader):
                with torch.no_grad():
                    real_dose_test = test_volumes[:, 0, :, :, :].unsqueeze(1).float()
                    ct_test = test_volumes[:, 1, :, :, :].unsqueeze(1).float()
                    prescription_test = test_volumes[:, 2, :, :, :].unsqueeze(1).float()
                    oars_test = test_volumes[:, 3, :, :, :].unsqueeze(1).float()

                    real_dose_test = real_dose_test.cuda()
                    oars_test = oars_test.cuda()
                    prescription_test = prescription_test.cuda()

                    model_input_test = torch.cat((ct_test, prescription_test, oars_test), dim=1)

                    y_fake_test = model(model_input_test)

                    loss_test = voxel_criterion(y_fake_test, real_dose_test)
                    cum_loss_test += loss_test.item()
                    test_loss_counter += 1
                    if test_loss_counter > 4:
                        break

            cum_loss_test /= test_loss_counter
                

        images_save_path = os.path.join(save_dir, "Images", exp_name_base, exp_name)
        if not os.path.exists(images_save_path):
            os.makedirs(os.path.join(images_save_path))

        # Saves images for all items in last batch
        if epoch % log_interval == 0:
            y_fake_test = y_fake_test.cpu().numpy()
            real_dose_test = real_dose_test.cpu().numpy()
            oars_test = oars_test.detach().cpu().numpy()
            ct_test = test_volumes[:, 1, :, :, :].unsqueeze(1).float().detach().numpy()
            for j in range(y_fake_test.shape[0]):
                plot = saveImgNby3(
                    [y_fake_test[j, 0, :, :, :], real_dose_test[j, 0, :, :, :]],
                    ct_test[j, 0, :, :, :],
                    os.path.join(images_save_path, str(j) + "_epoch" + str(epoch) + ".png"),
                    labels=["Fake", "Real"])
                writer.add_figure("Images from Testing Set " + str(j), plot, epoch)
                plt.close(plot)

        # Logging
        writer.add_scalar('loss_train', cum_loss, epoch)
        if epoch % log_interval == 0:
            writer.add_scalar('loss_test', cum_loss_test, epoch)


    writer.add_hparams(
        {"epochs": num_epochs, "loss_type": loss_type,
         "batch_size": batch_size, "g_lr": g_lr,
         },
        {"hparam/last_loss_test": cum_loss_test, },
        run_name=log_path)  # <- see here
    writer.close()

if __name__ == '__main__':
    print(torch.cuda.is_available())
    config = load_config("config.yml")
    data_dir = config.DATA_DIR
    patientList_dir = config.PATIENT_LIST_DIR
    save_dir = config.SAVE_DIR
    train_list_path = config.TRAIN_PATIENT_LIST
    test_list_path = config.TEST_PATIENT_LIST

    num_epochs = config.NUM_EPOCHS
    batch_size = config.BATCH_SIZE
    exp_name_base = config.EXP_NAME
    alphas = config.ALPHA
    betas = config.BETA
    loss_types = config.LOSS_TYPE
    g_lrs = config.G_LR
    d_lrs = config.D_LR

    adv_loss_types = config.ADV_LOSS_TYPE
    log_interval = config.LOG_INTERVAL


    runNum = 0
    for loss_type in loss_types:
        for g_lr in g_lrs:
            params = {
                "num_epochs": num_epochs,
                "loss_type": loss_type,
                "batch_size": batch_size,
                "g_lr": float(g_lr),
                "log_interval": log_interval,
            }
            exp_name = f'Unet_LR={g_lr}_Lo={loss_type}'
            print(params, exp_name)
            train(data_dir, train_list_path, test_list_path, save_dir,  exp_name_base, exp_name, params)

            runNum += 1