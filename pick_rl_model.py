import os
import argparse


""" Pick the best expert for PPO checkpoint """


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_folder', type=str, required=True)
    parser.add_argument('--limit_epoch', type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    best_acc = 0
    best_epoch_name = None
    epoch_folders = [f for f in os.listdir(args.checkpoint_folder) if os.path.isdir(os.path.join(args.checkpoint_folder, f))]
    epoch_folders.sort()
    for e in epoch_folders:
        parts = e.split('_')
        test_acc = float(parts[3])
        epoch = int(parts[1])
        if test_acc >= best_acc:
            best_acc = test_acc
            best_epoch_name = e
        if epoch >= args.limit_epoch:
            break
    print(os.path.join(args.checkpoint_folder, best_epoch_name))