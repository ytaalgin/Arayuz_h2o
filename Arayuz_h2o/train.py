import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
plt.ion()
class HLSDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ["Lake"]
        self.image_file_paths = []
        self.label_file_paths = []

        print(f"Exploring directory: {data_dir}")

        for class_name in self.classes:
            image_files = []
            label_files = []
            for root, _, subdirs in os.walk(data_dir):
                for subdir in subdirs:
                    subdir_path = os.path.join(root, subdir)
                    # Check if the subdirectory contains "B08.tif" files
                    image_files = glob.glob(os.path.join(subdir_path, "*.jpg"))
                    label_files = glob.glob(os.path.join(subdir_path, "*B09.tif"))
                    if ".jpg" in subdir_path:
                        image_files = glob.glob(subdir_path)
                    if "B09.tif" in subdir_path:
                        label_files = glob.glob(subdir_path) 
                    
                    # print(f"Subdirectory: {subdir_path}")
                    # print(f"Image files: {image_files}")
                    # print(f"Label files: {label_files}")

                    # Append image and label file paths
                    self.image_file_paths.extend(image_files)
                    self.label_file_paths.extend(label_files)
            self.label_encoder = LabelEncoder()
        # Map class names to class indices (0 for Lake)
        self.labels = self.label_encoder.fit_transform([self.classes[0] for _ in self.label_file_paths])

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        img_path = self.image_file_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        # label = self.labels[idx]
        label_path = self.label_file_paths[idx]

        # Use tifffile to open the TIFF label image and convert it to a NumPy ndarray
        label = tiff.imread(label_path)
        # Reduce the label to a single scalar value (e.g., by summing or averaging)
        label = np.sum(label) / (label.shape[0] * label.shape[1])  # Average the label values
        
        # Repeat the label to match the batch size (32) and convert it to a PyTorch tensor
        label = torch.tensor([label] * 1).float()  # Convert the label to a float

        
        
        # # Extract the corresponding label file path based on the image file path
        # label_path = self.label_file_paths[idx]
        
        # print(label_path)
        # # Use tifffile to open the TIFF label image and convert it to a NumPy ndarray
        # label = tiff.imread(label_path)

        
        # # Assuming you want to convert the label to a tensor, you can use transforms.ToTensor()
        # label = transforms.ToTensor()(label)
    
        return image, label

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

# Create an instance of the model
model = LakeDetectionModel()
# Define data transformations and preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}


# Define dataset directories for your data (modify these paths accordingly)
data_dir = "D:\Downloads\golet\S30\A10\S\G\J"

# Create custom dataset
dataset = HLSDataset(data_dir, transform=data_transforms['train'])
print(f"Number of samples in 2021 dataset: {len(dataset)}")

# Split the dataset into training and testing sets
train_ratio = 0.8  # You can adjust the ratio as needed
test_ratio = 1.0 - train_ratio

# Use train_test_split from scikit-learn
train_dataset, test_dataset = train_test_split(dataset, test_size=test_ratio, random_state=42)

# Create data loaders for training and testing
batch_size = 32  # Adjust the batch size as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Print the number of samples in each dataset to check if data is loaded


# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

fig, ax_loss = plt.subplots(figsize=(8, 6))  # Create a single subplot for loss
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Training Loss')
ax_loss.set_title('Training Loss History')


train_loss_history = []

num_epochs = 10
avg_loss = 0
# Training loop for 2021 data (you can add validation and more epochs)
try:
    while True:
        model.train()
        total_loss = 0.0  # Variable to accumulate the total loss for the epoch
        correct_predictions = 0
        total_predictions = 0  # Variable to accumulate the total loss for the epoch
        for epoch in range(num_epochs):
            for batch_idx, data in enumerate(train_dataloader):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)

                labels = labels.float()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Accumulate the batch loss
                total_loss += loss.item()
                
                # Print batch progress every, for example, every 10 batches
                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{10}], Batch [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item()}')

            # Calculate and print the average loss for the epoch
            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch [{epoch + 1}/{10}] Average Loss: {avg_loss}')
            # Append the average loss to the loss history list
            train_loss_history.append(avg_loss)
    
             # Compute the accuracy for the batch
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            batch_accuracy = 100 * correct_predictions / total_predictions
            # Append the batch loss and accuracy to the history lists
            train_loss_history.append(loss.item())

    
            # Update the real-time plots for loss and accuracy
            ax_loss.plot(range(1, len(train_loss_history) + 1), train_loss_history, marker='o')
            plt.pause(0.01)  # Pause to update the plots (adjust the interval as needed)



except KeyboardInterrupt:
    print("Training interrupted by the user. Saving the model.")
    # Save the trained model when the user interrupts training
    print(avg_loss)

    plt.savefig("training.png")
    # Disable interactive mode after training is complete
    plt.ioff()

    # Show the final plot (optional)
    plt.show()
    torch.save(model.state_dict(), 'water_level_model_interrupted.pth')

# Evaluate the model on the test dataset
model.eval()
total_rmse = 0.0
total_samples = 0
for data in test_dataloader:
    inputs, labels = data
    outputs = model(inputs)
    labels = labels.float()
    
    # Calculate RMSE for this batch
    batch_rmse = torch.sqrt(torch.mean((outputs - labels) ** 2)).item()
    
    total_rmse += batch_rmse
    total_samples += labels.size(0)

# Calculate the overall RMSE for the test dataset
overall_rmse = total_rmse / total_samples
print(f'Overall RMSE on the test dataset: {overall_rmse}')
