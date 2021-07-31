import yaml
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Function to load yaml configuration file
def load_config(config_name):
    with open(config_name) as file:
        config_f = yaml.safe_load(file)
    return config_f


def data_generator_with_aug(config):
    data_gen_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                           rescale=1. / 255.,
                                           horizontal_flip=config['pre_processing']['horizontal_flip'],
                                           rotation_range=config['pre_processing']['rotation_range'],
                                           zoom_range=config['pre_processing']['zoom_range'],
                                           zca_whitening=config['pre_processing']['zca_whitening'],
                                           width_shift_range=config['pre_processing']['width_shift_range'],
                                           height_shift_range=config['pre_processing']['height_shift_range'])
    return data_gen_with_aug


def data_generator_no_aug(config):
    data_gen_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                         rescale=1./255.)
    return data_gen_no_aug


def train_generator():
    config = load_config("config.yml")
    train_gen = data_generator_with_aug(config).flow_from_directory(
        config['pre_processing']['train_path'],
        target_size=(config['pre_processing']['image_size'], config['pre_processing']['image_size']),
        classes=config['pre_processing']['classes'],
        batch_size=config['pre_processing']['batch_size_train'],
        class_mode=config['pre_processing']['class_mode'],
        save_to_dir=config['pre_processing']['save_to_dir'],
        interpolation=config['pre_processing']['interpolation'])
    return train_gen


def validation_generator():
    config = load_config("config.yml")
    val_gen = data_generator_no_aug(config).flow_from_directory(
        config['pre_processing']['val_path'],
        target_size=(config['pre_processing']['image_size'], config['pre_processing']['image_size']),
        batch_size=config['pre_processing']['batch_size_test'],
        class_mode=config['pre_processing']['class_mode'],
        interpolation=config['pre_processing']['interpolation'])
    return val_gen
