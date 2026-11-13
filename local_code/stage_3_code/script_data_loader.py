import pickle
import os
from pathlib import Path
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2] #setting the wd to be ecs170stage2proj
os.chdir(PROJECT_ROOT)

# loading ORL dataset
if 1:
	f = open('data/stage_3_data/ORL', 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break
		
# loading CIFAR-10 dataset
if 0:
	f = open('data/stage_3_data/CIFAR', 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break

# loading MNIST dataset
if 0:
	f = open('data/stage_3_data/MNIST', 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix, cmap='gray')
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix, cmap='gray')
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break
	
	
	