import os
import sys
import random
import numpy as np
import json
import argparse
import pdb


# Arguments
parser = argparse.ArgumentParser(description="Create data path list")

parser.add_argument('--data_name', 
                            default='DEMAND',
                            type=str)

parser.add_argument('--data_root_dir', default='/home/leeji/home1/dataset/DEMAND/', type=str)
parser.add_argument('--save_dir', default='../data/preprocessed/', type=str)

parser.add_argument('--random_seed', default=1111, type=str)



def main(args):

    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Set data root directory
    data_root_split_dir = args.data_root_dir
    #-- data_root_split_dir: ex. /home/leeji/home1/dataset/DEMAND/

    # Search through data root directory
    path_list = []
    for (root_dir, sub_dir, files) in sorted(os.walk(data_root_split_dir)):
        for fname in sorted(files):
            ext = fname.split('.')[-1]
            if ext != 'wav':
                continue

            fpath = os.path.join(root_dir, fname)
            fpath = fpath.replace('/home/leeji/home1/dataset/DEMAND/', 'DEMAND/')
            # fpath: DEMAND/~~~/.wav
            path_list.append(fpath)

    # Save path list to json file
    output_name = args.data_name + '.json'      # ex. DEMAND.json
    output_path = args.save_dir + output_name

    # Save
    with open(output_path, 'w') as f:
        # indent=2 is not needed but makes the file human-readable 
        json.dump(path_list, f, indent=2) 

    print("FINISHED")



if __name__ == '__main__':

    args = parser.parse_args()
    print()

    # Check each arguments
    print("[ Arguments ]")
    print("data_name:       ", args.data_name)
    print("data_root_dir:   ", args.data_root_dir)
    print("save_dir:        ", args.save_dir)
    print()

    # Check
    answer = input("Write \033[31my\033[0m if you keep preprocessing: ")
    if answer != 'y':
        print("\nStop running...")
        sys.exit()

    # Run
    main(args)


