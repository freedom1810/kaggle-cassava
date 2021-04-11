import tqdm
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics
from torch.cuda import amp
import torch.nn.functional as F

from .augments import cutmix, snapmix

from .utils import freeze_model, unfreeze_model, save_checkpoint

import logging
logging.basicConfig(level=logging.INFO)


def trainer_augment(loaders, 
                    model_params, 
                    model, 
                    criterion, 
                    val_criterion, 
                    optimizer, 
                    lr_scheduler, 
                    optimizer_params, 
                    training_params, 
                    save_path):

    start_epoch = training_params['start_epoch']
    total_epochs = training_params['num_epoch']
    device = training_params['device']
    device_ids = training_params['device_ids']
    # augment_prob = model_params['special_augment_prob']

    model = nn.DataParallel(model, device_ids=device_ids)
    cuda = device.type != 'cpu'
    scaler = amp.GradScaler(enabled=cuda)
    ema = model_params['ema_model']

    print("Epochs: {}\n".format(total_epochs))
    best_epoch = 1
    best_acc = 0.0
    history = {
        "train": {"loss": [], "acc": []},
        "eval": {"loss": [], "acc": []},
        "lr": []
    }
    # num_layer = 213
    # num_layer = 340
    num_layer = 418
    for epoch in range(start_epoch, total_epochs + 1):
        if epoch <= training_params['warm_up'] and epoch == 1:
            training_params['TTA_time'] = 1
            model = freeze_model(model, num_layer)
            # print('-------------------------', ct)

        elif epoch == training_params['warm_up'] + 1:
            training_params['TTA_time'] = 5
            model = unfreeze_model(model)

        epoch_save_path = save_path + '_epoch-{}.pt'.format(epoch)
        head = "epoch {:2}/{:2}".format(epoch, total_epochs)
        print(head + "\n" + "-"*(len(head)))

        model.train()
        running_labels = 0
        running_scores = []
        running_loss = 0.0
        optimizer.zero_grad()
        for images, labels in tqdm.tqdm(loaders["train"]):
            images, labels = images.to(device), labels.to(device)
            with amp.autocast(enabled=cuda):

                outputs, _ = model(images, train_state=cuda)
                loss = criterion(outputs, labels)

                # snapmix_check = False
                # if np.random.rand(1) >= 2:
                #     snapmix_check = True
                #     SNAPMIX_ALPHA = 5.0
                #     mixed_images, labels_1, labels_2, lam_a, lam_b = snapmix(images, labels, SNAPMIX_ALPHA, model)
                #     mixed_images, labels_1, labels_2 = torch.autograd.Variable(mixed_images), torch.autograd.Variable(labels_1), torch.autograd.Variable(labels_2)
                    
                #     outputs, _ = model(mixed_images, train_state = cuda)
                #     loss_a = criterion(outputs, labels_1)
                #     loss_b = criterion(outputs, labels_2)
                #     loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
                #     running_labels += labels_1.shape[0]
                    
                # else:
                    
                #     mixed_images, labels_1, labels_2, lam = cutmix(images, labels)
                #     mixed_images, labels_1, labels_2 = torch.autograd.Variable(mixed_images), torch.autograd.Variable(labels_1), torch.autograd.Variable(labels_2)
                    
                #     outputs, _ = model(mixed_images, train_state = cuda)
                #     loss = lam*criterion(outputs, labels_1.unsqueeze(1)) + (1 - lam)*criterion(outputs, labels_2.unsqueeze(1))
                #     running_labels += labels_1.shape[0]

            # first step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if ema:
                ema.update(model)

            running_loss += loss.item()*images.size(0)

        epoch_loss = running_loss/running_labels
        history["train"]["loss"].append(epoch_loss)
        print("{} - loss: {:.4f}".format("train", epoch_loss))

        if ema:
            ema.update_attr(model, include=[])
            model_eval = ema.ema

        with torch.no_grad():
            if ema:
                model_eval = nn.DataParallel(model_eval, device_ids=device_ids)
                model_eval.eval()
            else:
                model.eval()

            final_scores = []
            final_loss = 0.0

            for TTA in range(training_params['TTA_time']):
                running_labels = []
                running_scores = []
                running_outputs_softmax = np.empty((0,model_params['num_classes']))
                running_loss = 0.0

                for images, labels in tqdm.tqdm(loaders["eval"]):
                    images, labels = images.to(device), labels.to(device)

                    if ema:
                        outputs, _ = model_eval(images, train_state=False)
                    else:
                        outputs, _ = model(images)

                    outputs_softmax = F.softmax(outputs, dim=-1)
                    scores = torch.argmax(outputs_softmax, 1)
                    loss = val_criterion(outputs, labels)
                    running_labels += list(labels.unsqueeze(1).data.cpu().numpy())
                    running_scores += list(scores.cpu().detach().numpy())
                    running_outputs_softmax = np.append(running_outputs_softmax, outputs_softmax.cpu().detach().numpy(), axis = 0)
                    running_loss += loss.item()*images.size(0)

                print("{} - TTA loss: {:.4f} acc: {:.4f}".format("eval", 
                        running_loss/len(running_labels), 
                        metrics.accuracy_score(running_labels, running_scores)))

                if TTA == 0:
                    final_scores = running_outputs_softmax
                else:
                    final_scores += running_outputs_softmax
                final_loss += running_loss/len(running_labels)

        running_labels_torch = torch.tensor(running_labels, dtype=torch.float32)
        epoch_loss = final_loss/training_params['TTA_time']

        final_scores = np.argmax(final_scores, axis=1)
        epoch_accuracy_score = metrics.accuracy_score(running_labels, final_scores)

        history["eval"]["loss"].append(epoch_loss.cpu().detach().numpy())
        history["eval"]["acc"].append(epoch_accuracy_score)

        print("{} loss: {:.4f} acc: {:.4f} lr: {:.9f}".format("eval - epoch",
                                                                epoch_loss, 
                                                                epoch_accuracy_score, 
                                                                optimizer.param_groups[0]["lr"]))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        lr_scheduler.step()

        if epoch_accuracy_score > best_acc:
            best_epoch = epoch
            best_acc = epoch_accuracy_score
        save_checkpoint(model_eval, optimizer, lr_scheduler, epoch, epoch_save_path)
        
    print("\nFinish: - Best Epoch: {} - Best accuracy: {}\n".format(best_epoch, best_acc))



    