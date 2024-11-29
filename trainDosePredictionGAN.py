import numpy as np
import os
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Custom
from models import Discriminator, AttentionGenerator
from dataset import VolumesFromList
from config import load_config
import matplotlib.pyplot as plt
from util import *

def getDLoss(g, d, real_dose, generator_input, adv_criterion):
    D_real = d(real_dose, generator_input)
    D_real_loss = adv_criterion(D_real, torch.randn_like(D_real) * 0.1 + 0.9) #Implementing one sided label smoothing
    y_fake = g(generator_input)
    D_fake = d(y_fake.detach(), generator_input)

    D_fake_loss = adv_criterion(D_fake, torch.zeros_like(D_fake))
    D_loss = (D_real_loss + D_fake_loss) / 2
    return D_loss, D_real_loss, D_fake_loss

def getGLoss(g, d, real_dose, generator_input, oars, adv_criterion, voxel_criterion, alpha, beta):
    y_fake = g(generator_input)
    D_fake = d(y_fake, generator_input)
    G_Dcomp_loss_train = adv_criterion(D_fake, torch.ones_like(D_fake))


    G_voxel_loss = voxel_criterion(y_fake, real_dose) / torch.numel(y_fake)
    G_masked_G_voxel_loss = voxel_criterion(y_fake * (oars > 0), real_dose * (oars > 0)) / torch.sum(oars > 0)
    G_loss = G_Dcomp_loss_train + alpha * ((1 - beta) * G_voxel_loss + beta * G_masked_G_voxel_loss)


    return G_loss, G_Dcomp_loss_train, G_voxel_loss, G_masked_G_voxel_loss, y_fake

def train(data_dir, train_list_path, test_list_path, save_dir, exp_name_base, exp_name, params):
    num_epochs = params["num_epochs"]
    alpha = params["alpha"]
    beta = params["beta"]
    log_interval = params["log_interval"]
    loss_type = params["loss_type"]
    batch_size = params["batch_size"]
    g_lr = params["g_lr"]
    d_lr = params["d_lr"]
    adv_loss_type = params["adv_loss_type"]

    g = AttentionGenerator(3, 1)

    g.cuda()
    d = Discriminator(in_features=4, last_conv_kernalsize=4)
    d.cuda()

    opt_g = optim.Adam(g.parameters(), lr=g_lr, betas=(0.5, 0.999), )
    opt_d = optim.Adam(d.parameters(), lr=d_lr, betas=(0.5, 0.999), )
    if adv_loss_type == "ls":
        adv_criterion = nn.MSELoss(reduction="mean").cuda()
    elif adv_loss_type == "bce":
        adv_criterion = nn.BCEWithLogitsLoss().cuda()
    #
    L2_LOSS_sum = nn.MSELoss(reduction="sum").cuda()

    if loss_type == "l1":
        voxel_criterion = nn.L1Loss(reduction="sum").cuda()
    elif loss_type == "l2":
        voxel_criterion = nn.MSELoss(reduction="sum").cuda()

    train_dataset = VolumesFromList(data_dir, train_list_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataset = VolumesFromList(data_dir, test_list_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    log_path = os.path.join(save_dir, "Logs", exp_name_base, exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    writer = SummaryWriter(log_path)

    epoch_loop = tqdm.tqdm(range(num_epochs + 1))
    # for epoch in range(num_epochs):
    for epoch in epoch_loop:
        g.train()
        d.train()
        for idx, volumes in enumerate(train_loader):


            real_dose = volumes[:, 0, :, :, :].unsqueeze(1).float()
            ct = volumes[:, 1, :, :, :].unsqueeze(1).float()
            prescription = volumes[:, 2, :, :, :].unsqueeze(1).float()
            oars = volumes[:, 3, :, :, :].unsqueeze(1).float()
            
            real_dose = real_dose.cuda()
            oars = oars.cuda()

            generator_input = torch.cat((ct, prescription, oars), dim=1)
            generator_input = generator_input.cuda()

            # Train Discriminator
            with torch.cuda.amp.autocast():
                D_loss, D_real_loss, D_fake_loss = getDLoss(g, d, real_dose, generator_input, adv_criterion)


            d.zero_grad()
            d_scaler.scale(D_loss).backward(retain_graph=True)
            d_scaler.step(opt_d)
            d_scaler.update()

            # Train Generator
            with torch.cuda.amp.autocast():
                #Note the y_fake that is returned has NOT been detached
                G_loss, G_Dcomp_loss_train, voxel_loss_train, masked_voxel_loss_train, y_fake = getGLoss(g, d, real_dose, generator_input, oars,
                                                                                               adv_criterion, voxel_criterion, alpha, beta)

            opt_g.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_g)
            g_scaler.update()

        # save weights
        weights_path = os.path.join(save_dir, "Weights", exp_name_base, exp_name)
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)

        if epoch % 100 == 0 or epoch == num_epochs:
            torch.save(g.state_dict(),
                       os.path.join(weights_path, "GeneratorWeightsEpoch" + str(epoch) + ".pth"))
            # torch.save(d.state_dict(),
            #            os.path.join(weights_path, "DiscriminatorWeightsEpoch" + str(epoch) + ".pth"))

        # calculate extra losses and metrics
        y_fake = y_fake.detach()
        train_mse_loss = L2_LOSS_sum(y_fake, real_dose) / torch.numel(y_fake)
        masked_train_mse_loss = L2_LOSS_sum(y_fake * (oars > 0), real_dose * (oars > 0)) / torch.sum(oars > 0)

        # Testing
        if epoch % log_interval == 0:
            g.eval()
            d.eval()
            G_loss_test = 0
            for test_idx, test_volumes in enumerate(test_loader):
                with torch.no_grad():
                    real_dose_test = test_volumes[:, 0, :, :, :].unsqueeze(1).float()
                    ct_test = test_volumes[:, 1, :, :, :].unsqueeze(1).float()
                    prescription_test = test_volumes[:, 2, :, :, :].unsqueeze(1).float()
                    oars_test = test_volumes[:, 3, :, :, :].unsqueeze(1).float()
                    
                    

                    generator_input_test = torch.cat((ct_test, prescription_test, oars_test), dim=1)

                    real_dose_test = real_dose_test.cuda()
                    generator_input_test = generator_input_test.cuda()
                    oars_test = oars_test.cuda()    

                    D_loss_test, D_real_loss_test, D_fake_loss_test = getDLoss(g, d, real_dose_test, generator_input_test, adv_criterion)

                    #Returned y_fake_test NOT been detached
                    #getGLoss(g, d, real_dose, generator_input, oars, adv_criterion, voxel_criterion, alpha, beta)
                    G_loss_test, G_Dcomp_loss_test, voxel_loss_test, masked_voxel_loss_test, y_fake_test = getGLoss(g, d,
                                                                                                           real_dose_test,
                                                                                                           generator_input_test,
                                                                                                            oars_test,
                                                                                                           adv_criterion,
                                                                                                           voxel_criterion, alpha, beta)

            G_loss_test /= (test_idx + 1)


            # calculate extra losses and metrics
            y_fake_test = g(generator_input_test).detach()
            test_mse_loss = L2_LOSS_sum(y_fake_test, real_dose_test) / torch.numel(y_fake_test)
            masked_test_mse_loss = L2_LOSS_sum(y_fake_test * (oars_test > 0), real_dose_test * (oars_test > 0)) / torch.sum(oars_test > 0)

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
        writer.add_scalar('LossG/train', G_loss, epoch)
        writer.add_scalar('LossG/test', G_loss_test, epoch)
        writer.add_scalar('LossD/train', D_loss, epoch)
        writer.add_scalar('LossD/test', D_loss_test, epoch)
        writer.add_scalar('LossD_fake/train', D_fake_loss, epoch)
        writer.add_scalar('LossD_real/train', D_real_loss, epoch)
        writer.add_scalar('LossD_fake/test', D_fake_loss_test, epoch)
        writer.add_scalar('LossD_real/test', D_real_loss_test, epoch)
        writer.add_scalar('MSE/train', train_mse_loss, epoch)
        writer.add_scalar('MSE/test', test_mse_loss, epoch)
        writer.add_scalar('Masked_MSE/train', masked_train_mse_loss, epoch)
        writer.add_scalar('Masked_MSE/test', masked_test_mse_loss, epoch)
        writer.add_scalar('G_D_loss/train', G_Dcomp_loss_train, epoch)
        writer.add_scalar('G_D_loss/test', G_Dcomp_loss_test, epoch)
        writer.add_scalar('G_Voxel_loss/train', voxel_loss_train, epoch)
        writer.add_scalar('G_Voxel_loss/test', voxel_loss_test, epoch)
        writer.add_scalar('Masked_G_Voxel_loss/train', masked_voxel_loss_train, epoch)
        writer.add_scalar('Masked_G_Voxel_loss/test', masked_voxel_loss_test, epoch)


    writer.add_hparams(
        {"epochs": num_epochs, "alpha": alpha, "beta": beta, "loss_type": loss_type,
         "batch_size": batch_size, "g_lr": g_lr, "d_lr": g_lr,
         "adv_loss_type": adv_loss_type},
        {"hparam/last_mse_loss_test": test_mse_loss, "hparam/last_g_loss_test": G_loss_test,
         "hparam/last_d_loss_test": D_loss_test},
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
    for alpha in alphas:
        for beta in betas:
            for loss_type in loss_types:
                    for g_lr in g_lrs:
                        for d_lr in d_lrs:
                            for adv_loss_type in adv_loss_types:
                                    params = {
                                        "num_epochs": num_epochs,
                                        "alpha": alpha,
                                        "beta": beta,
                                        "loss_type": loss_type,
                                        "batch_size": batch_size,
                                        "g_lr": float(g_lr),
                                        "d_lr": float(d_lr),
                                        "adv_loss_type": adv_loss_type,
                                        "log_interval": log_interval,
                                    }
                                    exp_name = f'dLR={d_lr}_gLR={g_lr}_A={alpha}_B={beta}_Lo={loss_type}'
                                    print(params, exp_name)
                                    train(data_dir, train_list_path, test_list_path, save_dir,  exp_name_base, exp_name, params)

                                    runNum += 1