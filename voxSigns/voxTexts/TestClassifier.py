import sys
import shutil
import os
from Classifier import Classifier


def create_output_dir(path, clear=False):
    if clear and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)


if len(sys.argv) < 5:
    print("too few command line arguments")
    sys.exit()

model_path = sys.argv[1]
classes_path = sys.argv[2]
test_data_dir = sys.argv[3]
out_data_dir = sys.argv[4]

print("Model path: ", model_path)
print("Classes path: ", classes_path)
print("Test data dir: ", test_data_dir)
print("Out data dir: ", out_data_dir)

classifier = Classifier.Classifier(model_file_path=model_path,
                                   category_to_id_mapping_file_path=classes_path)

print("Number of classes: ", classifier.get_number_of_classes())

create_output_dir(out_data_dir, clear=True)

dir_names = next(os.walk(test_data_dir))[1]
for dir_name in dir_names:

    in_dir_path = test_data_dir + dir_name + "/"
    print("handling class: ", dir_name)

    real_class_id = classifier.get_class_id_by_name(class_name=dir_name)
    if real_class_id is None:
        print("Couldn't find class index.")
        sys.exit()
        break;

    create_output_dir(out_data_dir + dir_name + "/")

    correct_prediction_num = 0
    total_prediction_num = 0
    file_names = next(os.walk(in_dir_path))[2]
    for file_name in file_names:

        file_path = in_dir_path + file_name

        class_id = classifier.classify_image_from_file(file_path)

        total_prediction_num += 1
        if class_id == real_class_id:
            correct_prediction_num += 1
        else:
            class_name = classifier.get_class_name_by_id(class_id)
            real_class_name = classifier.get_class_name_by_id(real_class_id)
            if (class_name is None) or (real_class_name is None):
                print("Couldn't get class name by id")
                sys.exit()

            class_out_data_dir = out_data_dir + class_name + "/"
            create_output_dir(class_out_data_dir)
            shutil.copyfile(src=file_path, dst=class_out_data_dir + real_class_name + '_' + file_name)

    print("correct prediction num: ", correct_prediction_num,
          ", total prediction num: ", total_prediction_num,
          ", acc: ", 1.0 * correct_prediction_num / total_prediction_num)
