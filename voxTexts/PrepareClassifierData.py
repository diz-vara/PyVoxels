import sys
import shutil
import os
import array
import random


def create_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def handle_class_dir(data_dir_path, class_dir_name, train_dir_path, validation_dir_path, test_dir_path):
    data_class_dir_path = data_dir_path + class_dir_name + '/'
    train_class_dir_path = train_dir_path + class_dir_name + '/'
    validation_class_dir_path = validation_dir_path + class_dir_name + '/'
    test_class_dir_path = test_dir_path + class_dir_name + '/'
    os.mkdir(train_class_dir_path)
    os.mkdir(validation_class_dir_path)
    os.mkdir(test_class_dir_path)
    files = next(os.walk(data_class_dir_path))[2]
    indexes = array.array('I')
    for i in range(0, len(files)):
        indexes.append(i)
    random.shuffle(indexes)
    train_file_number = 0
    for i in range(int(len(files) * 0.6)):
        file_name = files[indexes[i]]
        shutil.copyfile(data_class_dir_path + file_name, train_class_dir_path + file_name)
        train_file_number += 1
    validation_file_number = 0
    for i in range(int(len(files) * 0.6), int(len(files) * 0.8)):
        file_name = files[indexes[i]]
        shutil.copyfile(data_class_dir_path + file_name, validation_class_dir_path + file_name)
        validation_file_number += 1
    test_file_number = 0
    for i in range(int(len(files) * 0.8), len(files)):
        file_name = files[indexes[i]]
        shutil.copyfile(data_class_dir_path + file_name, test_class_dir_path + file_name)
        test_file_number += 1
    print("number of files: ", train_file_number, ', ', validation_file_number, ', ', test_file_number)


if len(sys.argv) < 2:
    print("too few command line arguments")
    sys.exit()

root_dir_path = sys.argv[1]
data_dir_path = root_dir_path + "data/"
train_dir_path = root_dir_path + "train/"
validation_dir_path = root_dir_path + "validation/"
test_dir_path = root_dir_path + "test/"
print("data dir: ", root_dir_path)

create_output_dir(train_dir_path)
create_output_dir(validation_dir_path)
create_output_dir(test_dir_path)

class_dir_names = next(os.walk(data_dir_path))[1]
print("number of classes: ", len(class_dir_names))

i = 0
for class_dir_name in class_dir_names:
    print("handling class #" + str(i) + ": " + class_dir_name)
    handle_class_dir(data_dir_path, class_dir_name, train_dir_path, validation_dir_path, test_dir_path)
    i += 1
