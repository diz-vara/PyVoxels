import sys
from TextClassifier import Trainer


if len(sys.argv) < 4:
    print("Too few command line arguments")
    sys.exit()

base_dir_path = sys.argv[1]
train_dir_path = base_dir_path + 'train/'
validation_dir_path = base_dir_path + 'validation/'
model_path = sys.argv[2]
class_to_id_mapping_path = sys.argv[3]
tensor_flow_dir_path = None
if len(sys.argv) >= 5:
    tensor_flow_dir_path = sys.argv[4]

print("train dir path: ", train_dir_path)
print("validation dir path: ", validation_dir_path)
print("model path: ", model_path)
print("class to id mapping path: ", class_to_id_mapping_path)
print("tensor flow dir path: ", tensor_flow_dir_path)

trainer = Trainer.Trainer(
    image_size=(32, 48),
    batch_size=16,
    number_of_epochs=100,
    learning_rate=0.0001)
number_of_train_samples = 12000

trainer.train_model(
    train_dir_path=train_dir_path,
    number_of_train_samples=number_of_train_samples,
    validation_dir_path=validation_dir_path,
    number_of_validation_samples=number_of_train_samples // 5,
    model_path=model_path,
    class_to_id_mapping_path=class_to_id_mapping_path,
    tensor_flow_log_dir_path=tensor_flow_dir_path)
