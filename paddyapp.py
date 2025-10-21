%%writefile paddyapp.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pickle

# Load class mapping
with open("class_to_idx.pkl", "rb") as f:
    class_to_idx = pickle.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Model definition (same as training)
class PaddyCNN(nn.Module):
    def __init__(self, num_classes):
        super(PaddyCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PaddyCNN(len(class_to_idx))
model.load_state_dict(torch.load("paddy_model.pth", map_location=device))
model.eval()

# Normalize function
def normalize_image(img_array):
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC ➜ CHW
    return torch.tensor(img_array).unsqueeze(0)

# Streamlit UI
st.title(" Paddy Leaf Disease Detector")
st.write("Upload a leaf image to predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((128, 128)).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediction
    img_array = np.array(image)
    tensor_img = normalize_image(img_array).to(device)

    with torch.no_grad():
        output = model(tensor_img)
        _, predicted = torch.max(output, 1)
        label = idx_to_class[predicted.item()]
        st.success(f" Predicted Disease: **{label}**")