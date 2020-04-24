import torch
import numpy as np
import torch.nn as nn

'''Todo: 
    1. find out how to compute loss: 
    output of enumerate(train_loader), 
    input of loss_fn
'''

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
        start_epoch=0, init = False):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    # for epoch in range(0, start_epoch):
    #     scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()
        '''
        UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. 
        In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  
        Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. 
        See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        '''

        # Train stage
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, init = False)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

        if val_loader is not None:
            val_loss = test_epoch(val_loader, model, loss_fn, cuda)
            val_loss /= len(val_loader)
            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, init = False):

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
            img_ab = tuple(d.cuda() for d in img_ab)
            if corres_ab is not None:
                # corres_ab = corres_ab.cuda()
                # modified
                corres_ab={key:corres_ab[key].cuda() for key in corres_ab}


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
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            # for metric in metrics:
            #     metric(outputs, target, loss_outputs)

    return val_loss