import pandas as pd
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# Dataset contains 200 bird species 
# submit a csv file with the image name and the prediced class label
# Order of the rows does not matter
# labels go from 1-200

# load data
train_set = pd.read_csv('train_images.csv')

test_set = pd.read_csv('test_images_sample.csv')
#print(test_set)

class_names = np.load("class_names.npy", allow_pickle=True).item()
#print(class_names)

# pre-processing - Example for the first image
first_image_path = train_set['image_path'][0]
image_example = Image.open(f"train_images{first_image_path}")

# resized/rescaled to the same resolution (224x224)
# normalized across the RGB channels with mean (0.5, 0.5, 0.5)
# and standard deviation (0.5, 0.5, 0.5)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
resized = processor.preprocess(images=image_example,
                                       do_resize=True,
                                       do_normalize=True,
                                       do_rescale=True,
                                       image_mean=[0.5, 0.5, 0.5],
                                       image_std=[0.5, 0.5, 0.5])

print(image_example)
print(resized)

