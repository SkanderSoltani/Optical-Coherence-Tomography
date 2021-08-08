import os
import numpy as np
import shutil
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
										 rescale=1. / 255.)
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


# Function to split the train dataset into new train/val sets
def split_data(source, destination, train_split=0.98, warm_start=False, warm_split=0.0):
	"""
	This function is used to split the train dataset into new train/val sets according to the parameters passed to
	function. It also allows the creation of a warm_start_data_split using just a warm_split % of the data.
	For consistency, please first create a new train/val split and then pass this new directory as source for
	the creation of a warm_start_data_split. In that way we guarantee that there's no cross-validation data been
	used in the warm_start phase of training.

	:param source: Path to the source folder where the original data is located
	:param destination: Path to the destination folder where the new dataset split will be copied to
	:param train_split: Ratio of the data that will go to the train folder of the new data split, the rest is copied to
    the val folder
	:param warm_start: If TRUE, the function will only copy warm_split % of the source data
	:param warm_split: The ratio of data to be used in the warm_start_data_split
	:return: None. Just copies the files.
	"""

	rng = np.random.default_rng()
	classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
	new_split_folder = "new_data_split"
	warm_start_folder = "warm_start_data_split"
	files_copied = zero_length_files = 0

	# If we are using warm_start than:
	# 1) Create the new folders (in case they don't exist) inside the "destination/warm_start_data_split" folder
	if warm_start:
		try:
			print("USING WARM START")
			print("Creating: ", os.path.join(destination, warm_start_folder), "\n")
			os.mkdir(os.path.join(destination, warm_start_folder))
			os.mkdir(os.path.join(destination, warm_start_folder, "train"))
			os.mkdir(os.path.join(destination, warm_start_folder, "val"))
		except OSError as err:
			print("OS error: {0}".format(err))
			pass
		# Update the destination path with the full warm_start_folder path
		destination = os.path.join(destination, warm_start_folder)

	# Create the folders in the "destination/new_data_split" otherwise
	else:
		try:
			print("USING REGULAR DATA SPLIT")
			print("Creating: ", os.path.join(destination, new_split_folder), "\n")
			os.mkdir(os.path.join(destination, new_split_folder))
			os.mkdir(os.path.join(destination, new_split_folder, "train"))
			os.mkdir(os.path.join(destination, new_split_folder, "val"))
		except OSError as err:
			print("OS error: {0}".format(err))
			pass
		# Update the destination path with the full new_split_folder path
		destination = os.path.join(destination, new_split_folder)

	# Use a for loop to go over the 4 classes into the train folder and split the data according
	# to the train_split size passed to the function
	for CLASS in classes:
		train_files = val_files = 0
		# Creating a folder for each class inside the new training and
		# validation directories of the destination.
		try:
			os.mkdir(os.path.join(destination + "/train", CLASS))
			os.mkdir(os.path.join(destination + "/val", CLASS))
		except OSError as err:
			print("OS error: {0}".format(err))
			pass

		# Create an array with the filenames from the train folder and insert the full path in the 2nd column
		random_source = np.asarray(os.listdir(os.path.join(source + "/train", CLASS)), dtype=object).reshape((-1, 1))
		random_source = np.insert(random_source, 1, os.path.join(source + "/train", CLASS), axis=1)
		print(f"The number of {CLASS} images in the source train folder is: ", len(random_source))

		# Shuffle the files
		rng.shuffle(random_source, axis=0)

		# Let's now start copying the files according to the train_split size
		for i, source_file in enumerate(random_source):
			# If we are using warm_start, than we just copy warm_split % of the data into
			# the training and validation folders
			if warm_start:
				if i < warm_split * train_split * len(random_source):
					# Checking to see if the file is of zero length
					if os.path.getsize(os.path.join(source_file[1], source_file[0])) != 0:
						shutil.copyfile(
							os.path.join(source_file[1], source_file[0]),
							os.path.join(destination + "/train", CLASS, source_file[0]),
						)
						files_copied += 1
						train_files += 1
					else:
						print(f"File {source_file[0]} is zero length so we don't copy it")
						zero_length_files += 1
				elif warm_split * train_split * len(random_source) <= i < warm_split * len(random_source):
					if os.path.getsize(os.path.join(source_file[1], source_file[0])) != 0:
						shutil.copyfile(
							os.path.join(source_file[1], source_file[0]),
							os.path.join(destination + "/val", CLASS, source_file[0]),
						)
						files_copied += 1
						val_files += 1
					else:
						print(f"File {source_file[0]} is zero length so we don't copy it")
						zero_length_files += 1

			# Not using warm_start. Splitting the whole data
			else:
				# Copying the training data
				if i < train_split * len(random_source):
					# Checking to see if the file is of zero length
					if os.path.getsize(os.path.join(source_file[1], source_file[0])) != 0:
						shutil.copyfile(
							os.path.join(source_file[1], source_file[0]),
							os.path.join(destination + "/train", CLASS, source_file[0]),
						)
						files_copied += 1
						train_files += 1
					else:
						print(f"File {source_file[0]} is zero length so we don't copy it")
						zero_length_files += 1
				# Copying the validation data
				else:
					if os.path.getsize(os.path.join(source_file[1], source_file[0])) != 0:
						shutil.copyfile(
							os.path.join(source_file[1], source_file[0]),
							os.path.join(destination + "/val", CLASS, source_file[0]),
						)
						files_copied += 1
						val_files += 1
					else:
						print(f"File {source_file[0]} is zero length so we don't copy it")
						zero_length_files += 1
		print(f"... copied {train_files} {CLASS} images to the new train set")
		print(f"... copied {val_files} {CLASS} images to the new val set\n")

	# Print the number of files copied to the new destination
	print("Total files copied: ", files_copied)
	if zero_length_files != 0:
		print(f"We found {zero_length_files} files with length zero the weren't copied")