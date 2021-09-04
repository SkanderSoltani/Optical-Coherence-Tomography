import os
import numpy as np
import shutil
import yaml
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load yaml configuration file
def load_config(config_name):
	with open(config_name) as file:
		config_f = yaml.safe_load(file)
	return config_f


class PreProcessing:
	def __init__(self):
		self._config = load_config('config.yml')
	

	def data_generator_with_aug(self):
		model_input = self._config['pre_processing']['input']
		datagen = ImageDataGenerator(preprocessing_function=preprocess_resnet if model_input == "resnet"
									else preprocess_inception if model_input == "inception"
									else preprocess_mobilenet if model_input == "mobilenet"
									else None,
									featurewise_center            = self._config['pre_processing']['featurewise_center'],
									samplewise_center             = self._config['pre_processing']['samplewise_center'],
									featurewise_std_normalization = self._config['pre_processing']['featurewise_std_normalization'],
									samplewise_std_normalization  = self._config['pre_processing']['samplewise_std_normalization'],
									zca_whitening                 = self._config['pre_processing']['zca_whitening'],
									zca_epsilon                   = self._config['pre_processing']['zca_epsilon'],
									rotation_range                = self._config['pre_processing']['rotation_range'],
									width_shift_range             = self._config['pre_processing']['width_shift_range'],
									height_shift_range            = self._config['pre_processing']['height_shift_range'],
									brightness_range              = self._config['pre_processing']['brightness_range'],
									shear_range                   = self._config['pre_processing']['shear_range'],
									zoom_range                    = self._config['pre_processing']['zoom_range'],
									channel_shift_range           = self._config['pre_processing']['channel_shift_range'],
									fill_mode                     = self._config['pre_processing']['fill_mode'],
									cval                          = self._config['pre_processing']['cval'],
									horizontal_flip               = self._config['pre_processing']['horizontal_flip'],
									vertical_flip                 = self._config['pre_processing']['vertical_flip'])
											
		return datagen


	def data_generator_no_aug(self):
		model_input = self._config['pre_processing']['input']
		datagen= ImageDataGenerator(preprocessing_function=preprocess_resnet if model_input == "resnet"
									else preprocess_inception if model_input == "inception"
									else preprocess_mobilenet if model_input == "mobilenet"
									else None)
		return datagen


	def getGenerator(self,path,augmentation_flag=True,test_gen=False):
		if augmentation_flag==True:
			data_generator = self.data_generator_with_aug()
			gen = data_generator.flow_from_directory(
				directory            = path,
				color_mode           = self._config['pre_processing']['color_mode'],
				batch_size           = self._config['pre_processing']['batch_size'],
				target_size          = (self._config['pre_processing']['target_size'],self._config['pre_processing']['target_size']),
				shuffle              = self._config['pre_processing']['shuffle'])
		elif test_gen:
			data_generator = self.data_generator_no_aug()
			gen = data_generator.flow_from_directory(
				directory            = path,
				color_mode           = self._config['pre_processing']['color_mode'],
				batch_size           = self._config['pre_processing']['batch_size'],
				target_size          = (self._config['pre_processing']['target_size'],self._config['pre_processing']['target_size']),
				shuffle              = False)
		else:
			data_generator = self.data_generator_no_aug()
			gen = data_generator.flow_from_directory(
				directory            = path,
				color_mode           = self._config['pre_processing']['color_mode'],
				batch_size           = self._config['pre_processing']['batch_size'],
				target_size          = (self._config['pre_processing']['target_size'],self._config['pre_processing']['target_size']),
				shuffle = self._config['pre_processing']['shuffle'])

		return gen 

	def getGenerators(self):
		train_path      = self._config['data']['train_path']
		val_path        = self._config['data']['val_path']
		test_path       = self._config['data']['test_path']
		warmup_path     = self._config['data']['warmup_path']
		
		# getting train generator:
		generator_train  = self.getGenerator(path=train_path,augmentation_flag=True)
		generator_val    = self.getGenerator(path=val_path,augmentation_flag=False)
		generator_test   = self.getGenerator(path=test_path,augmentation_flag=False,test_gen=True)
		generator_warmup = self.getGenerator(path=warmup_path,augmentation_flag=True)

		result = {"generator_train":generator_train,"generator_val":generator_val,"generator_test":generator_test,"generator_warmup":generator_warmup}
		return result



######################################################################################
#
#                 Data Split Training / Test
#
######################################################################################

	# Function to split the train dataset into new train/val sets
	def split_data(self,warm_start=False):
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
		# If using warm_start we use data from new_data_split as source
		if warm_start:
			source  = self._config['data']['source_warmup']

		# Otherwise we use the original dataset
		else:
			source  = self._config['data']['source']
		destination = self._config['data']['destination']
		train_split = self._config['pre_processing']['train_val_split']
		warm_split  = self._config['pre_processing']['warmup_split']

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
				os.mkdir(os.path.join(destination,"train", CLASS))
				os.mkdir(os.path.join(destination ,"val", CLASS))
			except OSError as err:
				print("OS error: {0}".format(err))
				pass

			# Create an array with the filenames from the train folder and insert the full path in the 2nd column
			random_source = np.asarray(os.listdir(os.path.join(source,"train", CLASS)), dtype=object).reshape((-1, 1))
			random_source = np.insert(random_source, 1, os.path.join(source,"train", CLASS), axis=1)
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
								os.path.join(destination,"train", CLASS, source_file[0]),
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
								os.path.join(destination,"val", CLASS, source_file[0]),
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
								os.path.join(destination,"train", CLASS, source_file[0]),
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
								os.path.join(destination,"val", CLASS, source_file[0]),
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