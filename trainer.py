import torch
import numpy as np
import torch.nn as nn
import os, copy
import matplotlib.pyplot as plt
from utils import save_checkpoint, get_lr
from tqdm import tqdm
# import wandb


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
        validation_frequency, save_root, init, 
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    # for epoch in range(0, start_epoch):
    # for epoch in range(1, start_epoch):
    #     scheduler.step()
    best_loss = 100000
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    val_x = []
    val_y = []
    val_y_contras = []
    val_y_gn = []


    train_x = []
    train_y = []
    train_y_contras = []
    train_y_gn = []


    for epoch in range(start_epoch, n_epochs):
        # scheduler.step()
        '''
        UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. 
        In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  
        Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. 
        See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        '''

        # Train stage
        train_loss, total_contras_loss, total_gnloss = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, save_root, epoch, init=False)
        train_x.append(epoch + 1)
        train_y.append(train_loss)
        train_y_contras.append(total_contras_loss)
        train_y_gn.append(total_gnloss)

        scheduler.step()
        message = '\nEpoch: {}/{}. Train set: Average loss: {:.4f}\ttriplet: {:.6f}\tgn loss: {:.6f}'.format(epoch + 1, n_epochs, train_loss, total_contras_loss, total_gnloss)
        message += ' Lr:{}'.format(get_lr(optimizer))
        if val_loader and (epoch % validation_frequency == 0):
            val_loss, val_contras_loss, val_gnloss = test_epoch(val_loader, model, loss_fn, cuda, epoch)
            val_loss /= len(val_loader)
            val_contras_loss /= len(val_loader)
            val_gnloss /= len(val_loader)


            val_x.append(epoch + 1)
            val_y.append(val_loss)
            val_y_contras.append(val_contras_loss)
            val_y_gn.append(val_gnloss)


            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}\ttriplet loss: {:.6f}\tgn loss: {:.6f}'.format(epoch + 1, n_epochs, val_loss, val_contras_loss, val_gnloss)
            # save the currently best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save_checkpoint(best_model_wts, True, save_root, str(epoch))
                message += '\nSaving best model ...'

        # save the model for every 20 epochs
        if (epoch % (n_epochs / 10)) == 0:
            message += '\nSaving checkpoint ... \n'
            save_checkpoint(model.state_dict(), False, save_root, str(epoch))
        print(message)

        # plt.figure(figsize=(12,8))
        # # plt.subplot(2, 2, 1)
        # plt.title("all_loss")
        # # plt.xlabel("epoch")
        # # plt.ylabel("loss_value")
        # plt.plot(val_x, val_y, "-s", label='val_total')
        # plt.plot(val_x, val_y_contras, "-o", label='val_contrastive')
        # plt.plot(val_x, val_y_gn, "-*", label='val_gn')
        # plt.plot(train_x, train_y, "+-", label='train_total')
        # plt.plot(train_x, train_y_contras, "-x", label='train_contrastive')
        # plt.plot(train_x, train_y_gn, "->", label='train_gn')
        # plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        # plt.savefig("./all_loss_pic.png")

        # plt.figure(figsize=(12,8))
        # plt.subplot(2, 1, 1)
        # plt.title("loss pos")
        # # plt.xlabel("epoch")
        # # plt.ylabel("loss_value")
        # plt.plot(val_x, val_y_loss_pos, "-s", label='val_total')
        # plt.plot(train_x, train_y_loss_pos, "+-", label='train_total')
        # plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

        # plt.subplot(2, 1, 2)
        # plt.title("loss neg")
        # # plt.xlabel("epoch")
        # # plt.ylabel("loss_value")
        # plt.plot(val_x, val_y_loss_neg, "-s", label='val_total')
        # plt.plot(train_x, train_y_loss_neg, "+-", label='train_total')
        # plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        # plt.savefig("./contrastive_loss_pic.png")


        plt.figure(figsize=(12,8))
        plt.subplot(2, 1, 1)
        plt.title("train_val_loss_pic")
        # plt.xlabel("epoch")
        # plt.ylabel("loss_value")
        plt.plot(val_x, val_y, "-s", label='val_total')
        plt.plot(train_x, train_y, "+-", label='train_total')
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        # plt.savefig("./train_val_loss_pic.png")

        plt.subplot(2, 2, 3)
        plt.title("triplet_loss")
        # plt.xlabel("epoch")
        # plt.ylabel("loss_value")
        plt.plot(val_x, val_y_contras, "-s", label='val_contrastive')
        plt.plot(train_x, train_y_contras, "+-", label='train_contrastive')
        # plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        # plt.savefig("./contrastive_loss_pic.png")

        plt.subplot(2, 2, 4)
        plt.title("gn_loss")
        # plt.xlabel("epoch")
        # plt.ylabel("loss_value")
        plt.plot(val_x, val_y_gn, "-s", label='val_gn')
        plt.plot(train_x, train_y_gn, "+-", label='train_gn')
        # plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        # plt.savefig("./gn_loss_pic.png")
        plt.savefig("./train_val_loss_pic.png")
        plt.close()




def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, save_root, epoch, init):
    # initialize network parameters, oscillates a lot here. not good
    if init:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    # Magic
    # wandb.watch(model)
    model.train()
    losses = []
    contras_losses = []
    gnlosses = []

    total_loss = 0
    total_contras_loss = 0
    total_gnloss = 0


    for batch_idx, (img_ab, corres_ab) in enumerate((train_loader)):
        corres_ab = corres_ab if len(corres_ab) > 0 else None
        if not type(img_ab) in (tuple, list):
            img_ab = (img_ab,)
        if cuda:
            img_ab = tuple(d.to(device) for d in img_ab)
            if corres_ab is not None:
                # corres_ab = corres_ab.cuda()
                # modified
                corres_ab = {key: corres_ab[key].to(device) for key in corres_ab}
                # modified by jzm
                # for c in corres_ab:
                #     c = {key: c[key].to(device) for key in c}

        optimizer.zero_grad()
        outputs = model(*img_ab)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        ''' the following needs modification
            change loss inputs
        '''
        if corres_ab is not None:
            corres_ab = (corres_ab,)
            loss_inputs += corres_ab
        # modified for triplet loss negative part
        loss_inputs += (epoch,)
        
        loss_outputs, contras_loss_outputs, gnloss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        contras_loss = contras_loss_outputs[0] if type(contras_loss_outputs) in (tuple, list) else contras_loss_outputs
        gnloss = gnloss_outputs[0] if type(gnloss_outputs) in (tuple, list) else gnloss_outputs


        losses.append(loss.item())
        contras_losses.append(contras_loss.item())
        gnlosses.append(gnloss.item())


        total_loss += loss.item()
        total_contras_loss += contras_loss.item()
        total_gnloss += gnloss.item()


        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttriplet_Loss: {:.6f}\tgn_Loss: {:.6f}'.format(
                batch_idx * len(img_ab[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses), np.mean(contras_losses), np.mean(gnlosses))

            print(message)
            losses = []
            contras_losses = []
            gnlosses = []


        del img_ab
        del corres_ab
        torch.cuda.empty_cache()
    total_loss /= (batch_idx + 1)
    total_contras_loss /= (batch_idx + 1)
    total_gnloss /= (batch_idx + 1)

    return total_loss, total_contras_loss, total_gnloss


def test_epoch(val_loader, model, loss_fn, cuda, epoch):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        val_contras_loss = 0
        val_gnloss = 0

        for batch_idx, (img_ab, corres_ab) in enumerate(val_loader):
            corres_ab = corres_ab if len(corres_ab) > 0 else None
            if not type(img_ab) in (tuple, list):
                img_ab = (img_ab,)
            if cuda:
                img_ab = tuple(d.to(device) for d in img_ab)
                if corres_ab is not None:
                    corres_ab = {key: corres_ab[key].to(device) for key in corres_ab}
                    # for c in corres_ab:
                    #     c = {key: c[key].to(device) for key in c}

            outputs = model(*img_ab)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if corres_ab is not None:
                corres_ab = (corres_ab,)
                loss_inputs += corres_ab

            # modified for triplet loss negative part
            loss_inputs += (epoch,)
            
            loss_outputs, contras_loss_outputs, gnloss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            contras_loss = contras_loss_outputs[0] if type(contras_loss_outputs) in (tuple, list) else contras_loss_outputs
            gnloss = gnloss_outputs[0] if type(gnloss_outputs) in (tuple, list) else gnloss_outputs

            val_loss += loss.item()
            val_contras_loss += contras_loss.item()
            val_gnloss += gnloss.item()


            # for metric in metrics:
            #     metric(outputs, target, loss_outputs)

    return val_loss, val_contras_loss, val_gnloss