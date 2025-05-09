
# 😷 Face Mask Detection and Notification System

A real-time computer vision application that detects whether individuals are wearing face masks. If a person is not wearing a mask for more than a specified duration, the system sends automated WhatsApp alerts using the Twilio API.

---

## 📌 Features

- ✅ Real-time face mask detection using OpenCV and a deep learning model.
- ✅ Sends WhatsApp alerts when someone is detected without a mask for over 5 or 10 seconds.
- ✅ Easy-to-integrate and run locally with webcam or IP camera support.
- ✅ Twilio integration for real-time remote alerts.

---

## 🛠️ Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- Twilio API
- Haar Cascade Classifier
- Pre-trained Mask Detection CNN

---

## 🔧 How It Works

1. Captures frames from webcam.
2. Detects faces using Haar cascades.
3. Classifies each face as:
   - Mask
   - No Mask
4. If “No Mask” detected for a continuous duration, a WhatsApp alert is triggered via Twilio.

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/vanshaggarwal27/Face-Mask-Detection-and-Notification
   cd Face-Mask-Detection-and-Notification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up `.env` file with your Twilio credentials:
   ```env
   TWILIO_ACCOUNT_SID=your_sid
   TWILIO_AUTH_TOKEN=your_token
   TWILIO_FROM=whatsapp:+14155238886
   TWILIO_TO=whatsapp:+91XXXXXXXXXX
   ```

4. Run the detection system:
   ```bash
   python main.py
   ```

---


## 📁 Folder Structure

```
Face-Mask-Detection-and-Notification/
│
├── main.py                  # Main script
├── mask_detector.model      # Trained CNN model
├── haarcascade_frontalface.xml
├── utils.py                 # Utility functions
├── .env                     # Environment variables
├── requirements.txt
├── client.jpg               # Sample output screenshot (client alert)
├── manager.jpg              # Sample output screenshot (manager alert)
└── README.md
```

---

## 🙋‍♂️ Author

**Vansh Aggarwal**  
GitHub: [@vanshaggarwal27](https://github.com/vanshaggarwal27)

---

## 📜 License

This project is licensed under the MIT License.
