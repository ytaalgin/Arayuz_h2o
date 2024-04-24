import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import sys
import os

class LakeDetectionModel(nn.Module):
    def __init__(self):
        super(LakeDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # Output 1 class (Water)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the water level detection model (replace 'YourWaterLevelModel' with your actual model class)
model = LakeDetectionModel()
model.load_state_dict(torch.load("water_level_model_interrupted.pth"))
model.eval()

img2018_path = sys.argv[1]
img2021_path = sys.argv[2]
# Determine the directory where your Python script is located
script_directory = os.path.dirname(os.path.abspath(__file__))


# Load and preprocess the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open(img2018_path)
image = transform(image)
image = image.unsqueeze(0)

image2 = Image.open(img2021_path)
image2 = transform(image2)
image2 = image2.unsqueeze(0)

# Make predictions using the model
with torch.no_grad():
    prediction_2018 = model(image).numpy()
    prediction_2021 = model(image2).numpy()

# Calculate the absolute pixel-wise difference between the two images
difference = cv2.absdiff(prediction_2018, prediction_2021)

# Sum up the pixel differences to get the total difference
total_difference = difference.sum().item()

# Create an image with text
texts = ['Oncesi', 'Sonrasi', f'Total Difference: {total_difference:}']
location = [(300, 50), (900, 50), (700, 520)]  # (x, y) coordinates

# Text settings
font_scale = 1
font_color = (0, 0, 0)  # BGR color format (black color)
font_thickness = 2

# Load the two images
image = cv2.imread(img2018_path)
image2 = cv2.imread(img2021_path)

width, height = 550, 550
image = cv2.resize(image, (width, height))
image2 = cv2.resize(image2, (width, height))

(thresh, blackAndWhiteImage) = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
# noinspection PyRedeclaration
(thresh, blackAndWhiteImage2) = cv2.threshold(image2, 15, 255, cv2.THRESH_BINARY)

# Set up the blob detector parameters
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 150  # Adjust the minimum area as needed

# The blob detector with the specified parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs in the inverted image
keypoints = detector.detect(blackAndWhiteImage)

# Draw blobs on the original image
image_with_blobs = cv2.drawKeypoints(blackAndWhiteImage, keypoints, np.array([]), (0, 0, 255),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image_with_blobs2 = cv2.drawKeypoints(blackAndWhiteImage2, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

combined_image = cv2.hconcat([image_with_blobs, image_with_blobs2])

for i, text in enumerate(texts):
    # Add the text to the image
    combined_image = cv2.putText(combined_image, text, location[i], cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color,
                                 font_thickness)
# Specify the output directory where you want to save the image
output_directory = sys.argv[3]  # The third argument contains the output image path

# Make sure the output directory exists; create it if necessary
os.makedirs(os.path.dirname(output_directory), exist_ok=True)

# Save the combined image to the specified output directory
output_image_path = os.path.join(os.getcwd(), output_directory)
print(output_image_path)
cv2.imwrite(output_image_path, combined_image)

# Show the image with detected blobs and predictions
cv2.imshow('Water Level Comparison', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
