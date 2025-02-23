import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import requests
import socketio
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from chatbot import query_knowledge_base, load_pdfs

st.set_page_config(page_title="AI Web App", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

inner_classes = ["Cataract", "Diabetic Retinopathy", "Glaucoma"]
outer_classes = ["Crossed Eyes", "Normal", "Uveitis"]

class EyeDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(EyeDiseaseCNN, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(model_path, num_classes):
    model = EyeDiseaseCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

inner_eye_model = load_model("inner_eye_model.pth", len(inner_classes))
outer_eye_model = load_model("outer_eye_model.pth", len(outer_classes))

def predict(image, model, class_labels):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_labels[predicted.item()]

sio = socketio.Client()
try:
    sio.connect("http://172.16.16.39:5000")
except:
    st.error("⚠️ Failed to connect to chat server!")

if "messages" not in st.session_state:
    st.session_state.messages = []

@sio.on("receive_message")
def receive_message(data):
    st.session_state.messages.append(f"{data['username']}: {data['message']}")
    st.experimental_rerun()

def main():
    st.title("AI Web App")
    menu = ["Home", "Register", "Login", "Predict", "Chatbot", "Webchat & Video Call"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("🏠 Welcome to the AI Web App")
    elif choice == "Register":
        st.subheader("📝 Register New User")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=1, max_value=120)
        address = st.text_area("Address")
        if st.button("Register"):
            data = {"username": username, "password": password, "name": name, "age": age, "address": address}
            response = requests.post("http://127.0.0.1:5000/register", json=data)
            st.success(response.json().get("message", "Error"))
    elif choice == "Login":
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            data = {"username": username, "password": password}
            response = requests.post("http://127.0.0.1:5000/login", json=data)
            st.success(response.json().get("message", "Error"))
    elif choice == "Predict":
        st.subheader("🔍 Eye Disease Detection")
        inner_eye_img = st.file_uploader("Upload Inner Eye Image", type=["png", "jpg", "jpeg"], key="inner")
        outer_eye_img = st.file_uploader("Upload Outer Eye Image", type=["png", "jpg", "jpeg"], key="outer")

        if st.button("Predict"):
            predictions = {}
            if inner_eye_img:
                image = Image.open(inner_eye_img).convert("RGB")
                st.image(image, caption="Uploaded Inner Eye Image", width=200)
                predictions["Inner Eye Disease"] = predict(image, inner_eye_model, inner_classes)
            else:
                st.warning("Upload Inner Eye image.")

            if outer_eye_img:
                image = Image.open(outer_eye_img).convert("RGB")
                st.image(image, caption="Uploaded Outer Eye Image", width=200)
                predictions["Outer Eye Disease"] = predict(image, outer_eye_model, outer_classes)
            else:
                st.warning("Upload Outer Eye image.")

            if predictions:
                st.subheader("🤓 Predictions")
                for key, value in predictions.items():
                    st.write(f"**{key}:** {value} ✅")
    elif choice == "Chatbot":
        st.subheader("🤖 Chatbot")
        if st.button("Load PDFs"):
            with st.spinner("Processing..."):
                load_pdfs()
            st.success("✅ PDFs Processed!")
        user_query = st.text_input("Ask a question:")
        if st.button("Ask"):
            if user_query.strip():
                response = query_knowledge_base(user_query)
                st.write("**Chatbot:**", response)
            else:
                st.warning("⚠ Enter a question.")
    elif choice == "Webchat & Video Call":
        st.subheader("📹 Video Call & Chatroom")
        username = st.text_input("Enter name:")
        room = st.text_input("Enter room ID:")
        if st.button("Join Room"):
            if username and room:
                sio.emit("join_room", {"username": username, "room": room})
                st.success(f"✅ Joined room {room} as {username}")
        st.subheader("💬 Chat Room")
        for msg in st.session_state.messages:
            st.write(msg)
        user_message = st.text_input("Type message:")
        if st.button("Send"):
            if sio.connected:
                sio.emit("send_message", {"username": username, "message": user_message, "room": room})
                st.session_state.messages.append(f"You: {user_message}")
                st.rerun()
            else:
                st.error("⚠ Not connected to chat server!")
        st.subheader("🎥 Video Call")
        webrtc_streamer(key="video", mode=WebRtcMode.SENDRECV, media_stream_constraints={"video": True, "audio": False})

if __name__ == "__main__":
    main()
