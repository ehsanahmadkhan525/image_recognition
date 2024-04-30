import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from scipy.spatial.distance import cosine

# Load a pre-trained model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define a transformation to resize and normalize the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.images = os.listdir(directory)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.directory, self.images[idx])
        image = Image.open(path).convert('RGB')
        return transform(image)

def get_embeddings(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for images in loader:
            features = model(images)
            embeddings.append(features)
    return torch.cat(embeddings)

def find_similar_images(upload_image_path, dataset):
    uploaded_image = Image.open(upload_image_path).convert('RGB')
    uploaded_image = transform(uploaded_image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        uploaded_features = model(uploaded_image).squeeze()  # Convert to 1-dimensional vector

    dataset_embeddings = get_embeddings(dataset)
    similarities = [1 - cosine(uploaded_features, emb.squeeze()) for emb in dataset_embeddings]  # Convert dataset embeddings to 1-dimensional vectors
    most_similar_idx = similarities.index(max(similarities))
    return dataset.images[most_similar_idx]

# Example usage
dataset = ImageDataset('dir')
similar_image = find_similar_images('img.jpg', dataset)
print(f"The most similar image is: {similar_image}")
