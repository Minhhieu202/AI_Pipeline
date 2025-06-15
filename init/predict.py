import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from ultralytics import YOLO 

class PlayerClassifier(nn.Module):
    def __init__(self, num_teams=2):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        del self.backbone.fc
        self.backbone.fc1 = nn.Linear(2048, 3)
        self.backbone.fc2 = nn.Linear(2048, 11)
        self.backbone.fc3 = nn.Linear(2048, num_teams)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.backbone.fc1(x)
        x2 = self.backbone.fc2(x)
        x3 = self.backbone.fc3(x)
        return x1, x2, x3


def inference():
    detector = YOLO("kaggle/working/runs/detect/train/weights/best.pt")  
    classifier = PlayerClassifier()
    checkpoint = torch.load("best.pt")
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    cap = cv2.VideoCapture("data/football_test\Match_1864_1_0_subclip\Match_1864_1_0_subclip.mp4")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (width, height))
    while cap.isOpened():
        flag, frame = cap.read()
        if flag:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            break
        detector_predictions = detector(frame)  # Run YOLOv11 detection
        player_coords = detector_predictions[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        player_images = []
        for coord in player_coords:
            xmin, ymin, xmax, ymax = coord
            player_image = frame[int(ymin): int(ymax), int(xmin): int(xmax), :]
            if player_image.size != 0:  # Ensure non-empty crop
                player_images.append(transform(player_image))
        
        if player_images:  # Only proceed if players are detected
            player_images = torch.stack(player_images)
            classifier_prediction = classifier(player_images)
            num_digits_prob, last_digits_prob, teams_prob = classifier_prediction
            num_digits = torch.argmax(num_digits_prob, dim=1)
            unit_digits = torch.argmax(last_digits_prob, dim=1)
            teams = torch.argmax(teams_prob, dim=1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for (xmin, ymin, xmax, ymax), n, u, t in zip(player_coords, num_digits, unit_digits, teams):
                u = u.item()
                if n == 2 or u == 10:
                    jersey_number = 0
                else:
                    if n == 0:
                        jersey_number = u
                    else:
                        jersey_number = 10 + u
                color = (255, 255, 255) if t == 0 else (0, 0, 0)
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(frame, str(jersey_number), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        out.write(frame)
    cap.release()
    out.release()


if __name__ == '__main__':
    inference()
