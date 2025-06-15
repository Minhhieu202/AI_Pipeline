import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image
class FootBallDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.image_paths = []
        self.labels = []
        self.categories = ["class0","class1","class2","class3","class4","class5",
                           "class6","class7","class8","class9","class10","class11"]
        self.transform = transform
        if train:
           data_path = os.path.join(root,"train")
        else:
           data_path = os.path.join(root,"test")
        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            for item in os .listdir(data_files):
                path = os.path.join(data_files,item)
                self.image_paths.append(path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        image = cv2.imread(img_path)  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = Image.fromarray(image) 
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    path = "football"
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    dataset = FootBallDataset(root = path, train = True, transform = transform)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = 16,
        shuffle = True,
        num_workers = 4,
        drop_last = True
    )

    for images, labels in dataloader:
        print(images.shape, labels.shape)

