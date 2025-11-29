import os
import random


def create_data_list(dataset_path, file_list, mode='train'):
    with open(os.path.join(dataset_path, (mode + '.txt')), 'w') as f:
        for (paths, _angle) in file_list:
            f.write(paths + ' ' + str(_angle) + '\n')
    print(mode + '.txt is created')

def get_file_list(directory, file_list, ext=None):
    new_dir = directory
    if os.path.isfile(directory):
        if ext is None:
            file_list.append(directory)
        else:
            if ext in directory[-3:]:
                file_list.append(directory)
    elif os.path.isdir(directory):
        for s in os.listdir(directory):
            new_dir = os.path.join(directory, s)
            get_file_list(new_dir, file_list, ext)
    return file_list

image_folder = 'data/images'
output_folder = 'data'
train_ratio = 0.8

jpg_list = get_file_list(image_folder, [], 'jpg')
print("Founded " + str(len(jpg_list)) + " jpg images in " + image_folder)

img_list = list()
for path in jpg_list:
    data_dir = os.path.dirname(path)
    base_name = os.path.basename(path)
    angle = (base_name[:-4]).split('_')[-1]
    img_path = os.path.join(data_dir, base_name).replace('\\', '/')
    img_list.append((img_path, angle))

random.seed(256)
random.shuffle(img_list)
train_num = int(len(img_list) * train_ratio)
tran_list = img_list[0:train_num]
test_list = img_list[train_num:]

create_data_list(output_folder, tran_list, 'train')
create_data_list(output_folder, test_list, 'test')