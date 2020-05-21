import os
import random
import argparse
import torch
from pprint import pprint
from torchvision.transforms import *
from utils import check_dir
from models.pretraining_backbone import ResNet18Backbone
from data.pretraining import DataReaderPlainImg
from sklearn.neighbors import NearestNeighbors
from data.pretraining import DataReaderPlainImg, custom_collate
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--weights-init', type=str,
                        default="")
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--output-root', type=str, default='results')
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(args.output_root, "nearest_neighbors",
                     args.weights_init.replace("/", "_").replace("models", "")))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # model
    #raise NotImplementedError("TODO: build model and load weights snapshot")
    # Device configuration
    data_root = args.data_folder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('#### This is the device used: ', device, '####')
    model = ResNet18Backbone(pretrained=False).to(device) 
    model.load_state_dict(torch.load(args.weights_init, map_location = device)['model'] , strict=False)
    # dataset
    val_transform = Compose([Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()])
    val_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2,
                                             pin_memory=True, drop_last=True, collate_fn=custom_collate)
    val_loader= val_loader.dataset
    #raise NotImplementedError("Load the validation dataset (crops), use the transform above.")
    print(val_loader)
    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    query_indices = [5]
    nns = []
    for idx, img in enumerate(val_loader):
        if idx not in query_indices:
            continue
        print("Computing NNs for sample {}".format(idx))
        closest_idx, closest_dist, images_names = find_nn(model, img, val_loader, 5, device)
        #raise NotImplementedError("TODO: retrieve the original NN images, save them and log the results.")


def find_nn(model, query_img, loader, k, device):
    """
    Find the k nearest neighbors (NNs) of a query image, in the feature space of the specified mode.
    Args:
         model: the model for computing the features
         query_img: the image of which to find the NNs
         loader: the loader for the dataset in which to look for the NNs
         k: the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    output_network = []
    loaded_data = torch.stack([x for i,x in enumerate(loader)])
    idx = [i for i in range(len(loaded_data)) if torch.all(loaded_data[i].eq(query_img))][0]
    list_images = loader.image_files
    with torch.no_grad():
        output_network = model(loaded_data)
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric = 'euclidean').fit(output_network.numpy())
    distances, indices = nbrs.kneighbors(output_network)

    #import pdb; pdb.set_trace()
    images_names = [ list_images[indices[idx][i]] for i in range(len(indices[idx]))]
    
    return indices[idx].tolist(), distances[idx].tolist(), images_names

    #raise NotImplementedError("TODO: nearest neighbors retrieval")
    # return closest_idx, closest_dist


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
