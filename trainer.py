import torch
import numpy as np
import torch.nn as nn
import os
from utils import save_checkpoint, get_lr

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")



def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, validation_frequency, save_root, init,
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
    best_loss = 0.0
    if not os.path.exists(save_root):
            os.makedirs(save_root)

    for epoch in range(start_epoch, n_epochs):
        # scheduler.step()
        '''
        UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. 
        In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  
        Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. 
        See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        '''

        # Train stage
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, save_root, init = False)
        scheduler.step()
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        message += ' Lr:{}'.format(get_lr(optimizer))
        if val_loader and (epoch % validation_frequency == 0):
            val_loss = test_epoch(val_loader, model, loss_fn, cuda)
            val_loss /= len(val_loader)
            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                            val_loss)
            # save the currently best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save_checkpoint(best_model_wts, True, save_root, str(epoch))
        
        # save the model for every 20 epochs
        if (epoch % log_interval/5) == 0:
            message += '\nSaving checkpoint ... \n'
            save_checkpoint(model.state_dict(), False, save_root, str(epoch))
        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, save_root, init):

    # initialize network parameters, oscillates a lot here. not good 
    if init:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (img_ab, corres_ab) in enumerate(train_loader):
        corres_ab = corres_ab if len(corres_ab) > 0 else None
        if not type(img_ab) in (tuple, list):
            img_ab = (img_ab,)
        if cuda:
            img_ab = tuple(d.to(device) for d in img_ab)
            if corres_ab is not None:
                # corres_ab = corres_ab.cuda()
                # modified
                corres_ab={key:corres_ab[key].to(device) for key in corres_ab}


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

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(img_ab[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)

    return total_loss


def test_epoch(val_loader, model, loss_fn, cuda):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (img_ab, corres_ab)  in enumerate(val_loader):
            corres_ab = corres_ab if len(corres_ab) > 0 else None
            if not type(img_ab) in (tuple, list):
                img_ab = (img_ab,)
            if cuda:
                img_ab = tuple(d.to(device) for d in img_ab)
                if corres_ab is not None:
                    corres_ab={key:corres_ab[key].to(device) for key in corres_ab}

            outputs = model(*img_ab)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if corres_ab is not None:
                corres_ab = (corres_ab,)
                loss_inputs += corres_ab

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            # for metric in metrics:
            #     metric(outputs, target, loss_outputs)

    return val_loss