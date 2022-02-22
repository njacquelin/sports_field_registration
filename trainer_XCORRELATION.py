import torch
from torch import save, load
from torch.nn import MSELoss, L1Loss, BCELoss, CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

import os
from dataloader import get_train_test_dataloaders

from model_deconv import vanilla_Unet2
from model import deeper_Unet_like, vanilla_Unet


if __name__=='__main__':
    torch.cuda.empty_cache()
    # writer = SummaryWriter('runs/training')

    train_img_path = '/path/to/Neptune Dataset/frames/train_all'
    out_path = './data_management/grids'
    lines_nb = 11
    model = vanilla_Unet2(final_depth=22).cuda()

    model_prefix = ''
    batch_size = 16
    models_path = './models/'
    epochs_already_trained = 0

    size = (256, 256)
    lr = 1e-3
    epochs_nb = 100

    optimizer_function = Adam
    save_after_N_epochs = 5
    display_frequency = 3

    initial_temperature = 1
    stagnation = 0.95

    train_dataloader, test_dataloader = get_train_test_dataloaders(train_img_path, out_path, size,
                                                  batch_size=batch_size, train_test_ratio=0.8, lines_nb=lines_nb)
    train_dataloader.temperature = initial_temperature
    test_dataloader.augment_data = False
    print('dataloader and model loaded')

    optimizer = optimizer_function(model.parameters(),
                                   lr=lr,
                                   weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_lr=1e-5, verbose=True)

    lines_criterion = CrossEntropyLoss(ignore_index=100)
    markers_criterion = CrossEntropyLoss(ignore_index=100)
    mask_criterion = BCELoss()
    mask_coef = 2
    markers_coef = 5

    if not os.path.isdir(models_path) : os.mkdir(models_path)

    if epochs_already_trained != 0:
        model.load_state_dict(load(models_path + model_prefix + 'best_model.pth'))

    display_counter = 0
    prev_best_loss = 1000

    for epoch in range(epochs_already_trained, epochs_already_trained + epochs_nb) :

        train_dataloader.temperature *= stagnation

        ### TRAIN PART ###
        total_epoch_loss = 0
        model.train()
        for batch in train_dataloader :
            img = batch['img'].cuda()
            truth = batch['out'].cuda()
            truth_mask = batch['mask'].cuda()

            out = model.forward(img)

            out_lines = out[:, :lines_nb]
            out_markers = out[:, lines_nb:]
            truth_lines = truth[:, 0]
            truth_markers = truth[:, 1]

            lines_loss = lines_criterion(out_lines, truth_lines)
            markers_loss = markers_criterion(out_markers, truth_markers)
            mask_loss = mask_criterion(torch.max(out, dim=1)[0], truth_mask)

            loss = lines_loss + \
                   markers_loss * markers_coef + \
                   mask_loss * mask_coef

            total_epoch_loss += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            display_counter += 1
            if display_counter == display_frequency :
                display_counter = 0
                print(float(loss))
        total_epoch_loss /= len(train_dataloader)
        print('train :', total_epoch_loss, epoch+1)

        ### TEST PART ###
        model.eval()
        total_epoch_loss = 0
        with torch.no_grad() :
            for batch in test_dataloader :
                img = batch['img'].cuda()
                truth = batch['out'].cuda()
                truth_mask = batch['mask'].cuda()

                out = model.forward(img)

                out_lines = out[:, :lines_nb]
                out_markers = out[:, lines_nb:]
                truth_lines = truth[:, 0]
                truth_markers = truth[:, 1]

                lines_loss = lines_criterion(out_lines, truth_lines)
                markers_loss = markers_criterion(out_markers, truth_markers)
                mask_loss = mask_criterion(torch.max(out, dim=1)[0], truth_mask)

                loss = lines_loss + \
                       markers_loss * markers_coef + \
                       mask_loss * mask_coef

                total_epoch_loss += float(loss)
            total_epoch_loss /= len(test_dataloader)

            print('test :', total_epoch_loss, epoch + 1)
            if total_epoch_loss < prev_best_loss:
                prev_best_loss = total_epoch_loss
                save(model.state_dict(), models_path + model_prefix + 'best_model.pth')
                print('\t\tSaved at epoch ' + str(epoch + 1))
            print()
        torch.cuda.empty_cache()

        scheduler.step(total_epoch_loss)

