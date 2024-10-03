import os
import shutil
from collections import defaultdict


def merge_dirs(dirs, dst):
    for dir in dirs:
        for item in os.listdir(dir):
            s = os.path.join(dir, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                merge_dirs([s], dst)  # Change this line
            else:
                if not os.path.exists(d):  # Check if file already exists
                    shutil.copy2(s, d)


def main():
    root_dir = "./data/drive_docs/"
    target_dir = "./data/drive_docs_v2/"

    dir_dict = defaultdict(list)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            dir_dict[dirname].append(os.path.join(dirpath, dirname))

    for dir_name, dirs in dir_dict.items():
        print(
            f"Directory name: {dir_name}, Count: {len(dirs)}"
        )  # Print directory name and count
        merge_target_dir = os.path.join(target_dir, dir_name)
        if not os.path.exists(merge_target_dir):
            os.makedirs(merge_target_dir)
        merge_dirs(dirs, merge_target_dir)


if __name__ == "__main__":
    main()
