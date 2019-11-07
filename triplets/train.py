from typing import Tuple
import copy
import os
import pickle
import time

import torch
import torch.utils


def train(
        model: torch.nn.Module,
        dataloaders: dict,
        criterion: torch.nn.Module,
        optimizer,
        scheduler: torch.optim.lr_scheduler,
        epochs: int,
        device: str,
        writer=None,
        model_name: str = 'base'
) -> Tuple[torch.nn.Module, list, list]:
    """
    Function to train model with given loss function, optimizer and scheduler. It operates in two phases: train and
    validation to allow to record results for validation set as well.
    :param model:
    :param dataloaders:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param epochs:
    :param device:
    :param writer:
    :param model_name:
    :return:
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 2e5
    total_loss_train, total_loss_val = [], []
    margin = criterion.margin
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_map = 0

            # Iterate over data.
            for idx, (data, labels) in enumerate(dataloaders[phase]):
                # Convert to tuple to avoid problems when unpacking value in model/loss forward call
                if not type(data) in (tuple, list):
                    data = (data,)
                data = tuple(d.to(device) for d in data)
                if len(labels) > 0:
                    labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(*data)
                    # Convert to tuple to avoid problems when unpacking value in model/loss forward call
                    if not type(outputs) in (tuple, list):
                        outputs = (outputs,)
                    if len(labels) > 0:
                        loss_outputs = criterion(*outputs, labels)
                    else:
                        loss_outputs = criterion(*outputs)
                    if type(loss_outputs) in (tuple, list):
                        loss, num_triplets = loss_outputs
                    else:
                        loss = loss_outputs
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * data[0].size(0)
                # running_map +=
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if writer:
                writer.add_scalar('Loss/train', epoch_loss, epoch)
            if phase == 'train':
                total_loss_train.append(epoch_loss)
            else:
                total_loss_val.append(epoch_loss)
                if epoch_loss < best_loss:
                    print("New best model found")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            # epoch_map = running_map.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

    if not os.path.exists('output/'):
        os.makedirs('output/')
    torch.save(model, f'output/model_{model_name}_margin_{margin}.pt')
    losses = {'train_loss': total_loss_train, 'val_loss': total_loss_val}
    with open(f'losses_model_{model_name}_margin_{margin}.pickle', 'wb') as f:
        pickle.dump(losses, f)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val mAP: {:4f}'.format(best_map))    # print('Best val mAP: {:4f}'.format(best_map))
    model.load_state_dict(best_model_wts)
    return model, total_loss_train, total_loss_val
