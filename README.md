# 🚗 Sleepy Driver Detection App 😴
_A real-time AI-powered drowsiness detection system_

## 📝 Overview
This is a real-time drowsiness detection application using **Streamlit** and **TensorFlow**. The app captures live video from a webcam, processes the frames, and predicts whether the driver's eyes are **open or closed** using a pre-trained deep learning model. If the driver is detected as **drowsy**, a warning message is displayed. ⚠️

## 🌟 Features
✅ **Live** webcam feed processing using **OpenCV**  
✅ **Real-time** drowsiness detection with AI  
✅ **Deep learning** model for eye state classification  
✅ **User-friendly** Streamlit interface  
✅ **Alerts and warnings** when drowsiness is detected  

---
## 🛠 Requirements
Ensure you have the following installed before running the application:

### 📦 Python Packages
```bash
pip install streamlit tensorflow opencv-python numpy
```

### 💻 Hardware Requirements
- 🎥 **Webcam** for live video capture
- 🖥 **Computer with TensorFlow-compatible hardware** (GPU recommended but not required)

---
## 🚀 Installation & Usage

1. 📥 **Clone the repository:**
```bash
git clone https://github.com/AdityaTagde/SleepSafeDrive.git
cd sleepy-driver-detection
```

2. 📦 **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. 📂 **Place your trained model (`my_model.h5`) in the project directory.**

4. ▶️ **Run the application:**
```bash
streamlit run app.py
```

5. ✅ **Start live detection** by checking the checkbox in the app.

---
## 🧠 Model Details
The deep learning model (`my_model.h5`) is a classification model trained to detect **eye states:**
- **😴 Closed** (drowsy state)
- **😃 Opened** (awake state)

📏 **Model Input:** Images are resized to `180x180` pixels for prediction.

---
## 🔍 How It Works
1. 📸 **Captures frames** from the webcam.
2. 🖼 **Preprocesses** the image (resized, converted to an array, normalized, etc.).
3. 🧠 **Passes the frame through a trained AI model** for classification.
4. ⚠️ **Displays a warning** if drowsiness is detected.
5. ✅ **Shows a green "Awake" message** if the driver is alert.
6. 📺 **Live visualization** of processed frames in the Streamlit UI.

---
## 🛠 Troubleshooting
- ❌ **Webcam not working?** Ensure it's not being used by another application.
- ❌ **Model not loading?** Check that `my_model.h5` is correctly placed in the directory.
- 🔄 **Getting package errors?** Try updating dependencies:
```bash
pip install --upgrade tensorflow opencv-python streamlit numpy
```

---
## 🔮 Future Improvements
✨ **Add sound alerts** for drowsiness detection. 🔊  
✨ **Improve accuracy** with a larger dataset and better model architecture. 📊  
✨ **Deploy as a web application** for mobile compatibility. 📱  

### 📸 App Demo
![App View](https://github.com/AdityaTagde/SleepSafeDrive-/blob/main/s1.png)

