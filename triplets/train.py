import copy
import os
import time

import torch


def train(model, dataloaders, criterion, optimizer, scheduler, epochs, device, writer=None, model_name='base'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 2e5
    total_loss_train, total_loss_val = [], []
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
                data = tuple(d.to(device) for d in data)
                if labels:
                    labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(*data)
                    loss = criterion(*outputs)
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
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            # epoch_map = running_map.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            if not os.path.exists('output/'):
                os.makedirs('output/')
            torch.save(model, f'output/model_{model_name}_epoch_{epoch}.pt')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val mAP: {:4f}'.format(best_map))    # print('Best val mAP: {:4f}'.format(best_map))
    model.load_state_dict(best_model_wts)
    return model
