raise NotImplementedError("TODO: make the script for running the classical multiclass semantic segmentation")
# Use the dt_binary_ss as a starting point for your file
# start you lr at 0.001, try more if you have time
# Use DataReaderSemanticSegmentation as dataloader
# Use utils.instance_mIoU to evaluate your model


import os
from pprint import pprint
import random
import sys
sys.path.insert(0,os.getcwd())
import numpy as np
import argparse
import torch
import time
from utils import check_dir, set_random_seed, accuracy, instance_mIoU, get_logger
from models.second_segmentation import Segmentator
from data.transforms import get_transforms_binary_segmentation
from models.pretraining_backbone import ResNet18Backbone
from data.segmentation import DataReaderSemanticSegmentation
import pandas as pd
import matplotlib.pyplot as plt

set_random_seed(0)
global_step = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data")
    parser.add_argument('weights_init', type=str, default="ImageNet")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--size', type=int, default=256, help='image size')
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'dt_binseg', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))
    args.plots_folder = check_dir(os.path.join(args.output_folder, "plots"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)
    img_size = (args.size, args.size)

    # model
    data_root = args.data_folder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('#### This is the device used: ', device, '####')
    import pdb; pdb.set_trace()
    pretrained_model = ResNet18Backbone(pretrained=False).to(device) 
    #pretrained_model = None
    #raise NotImplementedError("TODO: build model and load pretrained weights")
    model = Segmentator(6, pretrained_model.features, img_size).to(device) # 5 + background

    #features = list(vgg16(pretrained = True).features)[:23]
    #model.load_state_dict(torch.load(args.weights_init, map_location = device)['model'], strict=False)

    # dataset
    train_trans, val_trans, train_target_trans, val_target_trans = get_transforms_binary_segmentation(args)
    data_root = args.data_folder
    train_data = DataReaderSemanticSegmentation(
        os.path.join(data_root, "imgs/train2014"),
        os.path.join(data_root, "aggregated_annotations_train_5classes.json"),
        transform=train_trans,
        target_transform=train_target_trans
    )
    val_data = DataReaderSemanticSegmentation(
        os.path.join(data_root, "imgs/val2014"),
        os.path.join(data_root, "aggregated_annotations_val_5classes.json"),
        transform=val_trans,
        target_transform=val_target_trans
    )
    print("Dataset size: {} samples".format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True,
                                               num_workers=6, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,
                                             num_workers=6, pin_memory=True, drop_last=False)

    # TODO: loss
    criterion = torch.nn.CrossEntropyLoss() # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    # TODO: SGD optimizer (see pretraining)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    #raise NotImplementedError("TODO: loss function and SGD optimizer")

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    best_val_miou = 0.0

    train_loss_list = []
    train_iou_list = []
    val_loss_list = []
    val_iou_list = []
    for epoch in range(100):
        logger.info("Epoch {}".format(epoch))
        train_results = train(train_loader, model, criterion, optimizer, logger, device)
        val_results = validate(val_loader, model, criterion, logger, device, epoch)

        mean_val_loss, mean_val_iou = val_results
        mean_train_loss, mean_train_iou = train_results

        train_loss_list.append(mean_train_loss)
        train_iou_list.append(mean_train_iou)
        val_loss_list.append(mean_val_loss)
        val_iou_list.append(mean_val_iou)

        # TODO save model
        save_model(model, optimizer, args, epoch, mean_val_loss, mean_train_loss, logger, best=False)
        # path_model = os.path.join(args.model_folder , 'checkpoint_' + str(epoch) +'_.pth')
        # torch.save(model.state_dict(), path_model )
        # import pdb; pdb.set_trace()
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            save_model(model, optimizer, args, epoch, mean_val_loss, mean_train_loss, logger, best=True)
        
        # save the data
        save_fig (train_loss_list, 'train_loss')
        save_fig (train_iou_list, 'train_acc')
        save_fig (val_loss_list, 'val_loss')
        save_fig (val_iou_list, 'val_acc')

        pd.DataFrame({'train_loss':train_loss_list}).to_csv(os.path.join(args.plots_folder, 'train_loss.csv'), index= False)
        pd.DataFrame({'train_iou':train_iou_list}).to_csv(os.path.join(args.plots_folder, 'train_iou.csv'), index= False)
        pd.DataFrame({'val_loss':val_loss_list}).to_csv(os.path.join(args.plots_folder, 'val_loss.csv'), index= False)
        pd.DataFrame({'val_iou':val_iou_list}).to_csv(os.path.join(args.plots_folder, 'val_iou.csv'), index= False)

        



def train(loader, model, criterion, optimizer, logger, device):
    train_loss = 0
    train_iou = 0
    for batch_i, (data, target) in enumerate(loader):
        target = target * 255
        target = torch.squeeze(target,1).long()

        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_iou += mIoU(output, target.to(device)).item()

        if ((batch_i % 150)== 0):
            print('train batch ', batch_i, ' with loss: ', round(train_loss/(batch_i+1),5), ' with iou: ', round(train_iou/(batch_i+1),5))

    mean_train_loss = round((train_loss/(batch_i+1)), 5)
    mean_train_iou = round((train_iou/(batch_i+1)), 5)

    logger.info("mean train_loss {} mean train_iou {}".format(mean_train_loss, mean_train_iou))

    return (mean_train_loss, mean_train_iou)
    #raise NotImplementedError("TODO: training routine")


def validate(loader, model, criterion, logger, device, epoch=0):
    model.eval()
    val_loss = 0
    val_iou = 0
    with torch.no_grad():
        for batch_i, (data, target) in enumerate(loader):
            output = model(data.to(device))

            target = target * 255
            target = torch.nn.functional.interpolate(target, ( output.shape[2], output.shape[3]) )
            target = torch.squeeze(target,1).long()

            loss = criterion(output, target.to(device))
            val_loss += loss.mean().item()
            val_iou += mIoU(output, target.to(device)).item()
            if ((batch_i % 150)== 0):
                print('val batch ', batch_i, ' with loss: ', round(val_loss/(batch_i+1),5), ' with iou: ', round(val_iou/(batch_i+1),5))

    mean_val_loss = round((val_loss/(batch_i+1)), 5)
    mean_val_iou = round((val_iou/(batch_i+1)), 5)

    logger.info("epoch {} mean val_loss {} mean val_iou {}".format(epoch, mean_val_loss, mean_val_iou))
    #raise NotImplementedError("TODO: validation routine")
    return (mean_val_loss, mean_val_iou)


def save_model(model, optimizer, args, epoch, val_loss, val_iou, logger, best=False):
    # save model
    add_text_best = 'BEST' if best else ''
    logger.info('==> Saving '+add_text_best+' ... epoch {} loss {:.03f} miou {:.03f} '.format(epoch, val_loss, val_iou))
    state = {
        'opt': args,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': val_loss,
        'miou': val_iou
    }
    if best:
        torch.save(state, os.path.join(args.model_folder, 'ckpt_best.pth'))
    else:
        torch.save(state, os.path.join(args.model_folder, 'ckpt_epoch {}_loss {:.03f}_miou {:.03f}.pth'.format(epoch, val_loss, val_iou)))


def save_fig (train_list, name):
    plt.plot(train_list)
    plt.xlabel('epochs')
    plt.ylabel(name)
    if not os.path.exists(args.plots_folder):
        os.makedirs(args.plots_folder)
    path = os.path.join(args.plots_folder, name+'.png')
    plt.savefig(path)

if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
