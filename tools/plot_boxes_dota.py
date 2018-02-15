import numpy as np
import os
import cv2

dataset_dir = '/home/victore/Pictures/DOTA/train'
N = 10

input_images = sorted([os.path.join(dataset_dir, 'images', file) for file in
                       os.listdir(dataset_dir + '/images') if file.endswith('.png')])[:N]

input_labels = sorted([os.path.join(dataset_dir, 'labelTxt', file) for file in
                       os.listdir(dataset_dir + '/labelTxt') if file.endswith('.txt')])[:N]

input_pairs = zip(input_images, input_labels)

output_dir = os.path.join(dataset_dir, 'labelImages')
os.makedirs(output_dir, exist_ok=True)

for (image_file, label_file) in input_pairs:
    boxes = np.loadtxt(label_file, skiprows=2, usecols=tuple(range(8)))
    img = cv2.imread(image_file)
    for row in range(boxes.shape[0]):
         cv2.polylines(img, [np.int32(boxes[row].reshape(4,1,2))], isClosed=True, thickness=3, color=(255,0,0))
    cv2.imwrite(os.path.join(output_dir, os.path.basename(image_file)), img)



