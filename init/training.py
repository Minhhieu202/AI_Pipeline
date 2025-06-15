import torch
from dataset import FootBallDataset
from model import MYCNN
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.metrics import accuracy_score  

def train():
    path = "football"
    batch_size = 16
    learning_rate = 1e-3
    momentum = 0.9
    num_epochs = 150

    # Kiểm tra GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = MYCNN(num_classes=12).to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize trước khi ToTensor
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5], [0.5])  
    ])

    train_dataset = FootBallDataset(root=path, train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    val_dataset = FootBallDataset(root=path, train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader)
        num_iter = len(train_dataloader)
        for iter, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)  
            optimizer.zero_grad()  
            outputs = model(images)  
            loss = criterion(outputs, labels) 
            progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

            # Backpropagation
            loss.backward() 
            optimizer.step() 
        
        # Validation
        model.eval()
        val_losses = []
        val_labels = []
        val_predictions = []
        progress_bar = tqdm(val_dataloader, colour="red")
        for iter, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)  
            with torch.no_grad():
                outputs = model(images)  
            predictions = torch.argmax(outputs, dim=1) 
            loss = criterion(outputs, labels) 
            
            val_labels.append(labels.cpu())  # back to  CPU
            val_losses.append(loss.item()) 
            val_predictions.append(predictions.cpu())

        # Chuyển danh sách tensor thành numpy arrays
        val_labels = torch.cat(val_labels).numpy()
        val_predictions = torch.cat(val_predictions).numpy()

        avg_loss = np.mean(val_losses)
        acc = accuracy_score(val_labels, val_predictions)
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {acc:.4f}")

        if acc > best_acc:
            torch.save({'state_dict': model.state_dict()}, "player_classification.pt")
            best_acc = acc
if __name__ == '__main__':
    train()
