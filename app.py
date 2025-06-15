import streamlit as st
import cv2
import tempfile
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
import numpy as np
from PIL import Image

# --- Model Definition ---
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


# --- Inference Function ---
def run_inference(input_path, output_path, classifier, detector, transform):
    cap = cv2.VideoCapture(input_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"),
                          int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector_predictions = detector(rgb_frame)
        player_coords = detector_predictions[0].boxes.xyxy.cpu().numpy()

        player_images = []
        coords = []
        for coord in player_coords:
            xmin, ymin, xmax, ymax = coord
            player_img = rgb_frame[int(ymin): int(ymax), int(xmin): int(xmax)]
            if player_img.size != 0:
                img_tensor = transform(player_img)
                player_images.append(img_tensor)
                coords.append((xmin, ymin, xmax, ymax))

        if player_images:
            with torch.no_grad():
                player_images = torch.stack(player_images)
                num_digits_prob, last_digits_prob, teams_prob = classifier(player_images)
                num_digits = torch.argmax(num_digits_prob, dim=1)
                unit_digits = torch.argmax(last_digits_prob, dim=1)
                teams = torch.argmax(teams_prob, dim=1)

            for (xmin, ymin, xmax, ymax), n, u, t in zip(coords, num_digits, unit_digits, teams):
                u = u.item()
                if n == 2 or u == 10:
                    jersey_number = 0
                else:
                    jersey_number = u if n == 0 else 10 + u
                color = (255, 255, 255) if t == 0 else (0, 0, 0)
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(frame, str(jersey_number), (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()


# --- Streamlit App ---
st.title("Football Player Detection & Classification")
uploaded_file = st.file_uploader("Upload a football match video", type=["mp4", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_video_path = temp_input.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as temp_output:
        output_video_path = temp_output.name

    st.write("⏳ Processing video...")

    # Load models
    detector = YOLO("player_detector.pt")  # adjust if needed
    classifier = PlayerClassifier()
    checkpoint = torch.hub.load("player_classification.pt")
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    run_inference(input_video_path, output_video_path, classifier, detector, transform)

    st.success("✅ Video processing complete!")
    st.video(output_video_path)
    with open(output_video_path, "rb") as file:
        st.download_button(label="Download Processed Video", data=file, file_name="processed_output.avi")
