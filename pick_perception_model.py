import os
import argparse
import json


""" Pick the best perception model """


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_folder', type=str, required=True)
    parser.add_argument('--limit_epoch', type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    best_error_dist = float('inf')
    best_epoch_name = None
    epoch_folders = [f for f in os.listdir(args.checkpoint_folder) if os.path.isdir(os.path.join(args.checkpoint_folder, f))]
    epoch_folders.sort()
    for e in epoch_folders:
        data = json.load(open(os.path.join(args.checkpoint_folder, e, 'checkpoint_metadata.json')))
        train_loss = data['train_epoch_loss']
        test_loss = data['test_epoch_loss']
        train_error_dist = data['train_epoch_error_distance']
        test_error_dist = data['test_epoch_error_distance']
        epoch = data['epoch']
        if test_error_dist <= best_error_dist:
            best_error_dist = test_error_dist
            best_epoch_name = e
        if epoch >= args.limit_epoch:
            break
    print(os.path.join(args.checkpoint_folder, best_epoch_name))