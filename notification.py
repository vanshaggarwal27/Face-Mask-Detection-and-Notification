import cv2
import numpy as np
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cvlib as cv
from twilio.rest import Client

# ─── Twilio WhatsApp Setup ──────────────────────────────────────────────────────
# 1) pip install twilio
# 2) Sign up at twilio.com → get ACCOUNT_SID, AUTH_TOKEN, a WhatsApp-enabled number
TWILIO_SID   = 'Your_account_ssid'
TWILIO_TOKEN = 'Your_account_token'
WHATSAPP_FROM = 'whatsapp:+14155238886'    # your Twilio sandbox number
USER_WHATSAPP = 'whatsapp:+91XXXXXXXXXX'  # the user’s WhatsApp number
MANAGER_WHATSAPP = 'whatsapp:+91XXXXXXXXXX'

twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

def send_whatsapp(to, body):
    message = twilio_client.messages.create(
        body=body,
        from_=WHATSAPP_FROM,
        to=to
    )
    print(f"Sent WhatsApp to {to}: SID {message.sid}")

# ─── Load Models ────────────────────────────────────────────────────────────────
model = load_model('mask_detection_model.h5')
cap   = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ─── Per-face Tracking State ───────────────────────────────────────────────────
# We'll identify faces by their bounding-box centroids (simple approach).
no_mask_timers = {}  # key: face_id, value: time when no-mask first seen

def face_id_from_box(box):
    x1,y1,x2,y2 = box
    cx, cy = (x1+x2)//2, (y1+y2)//2
    return (cx//50, cy//50)  # bucket into 50-px grid for crude tracking

# ─── Main Loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces, _ = cv.detect_face(frame)
    current = set()

    for box in faces:
        x1,y1,x2,y2 = box
        current_id = face_id_from_box(box)
        current.add(current_id)

        face_img = frame[y1:y2, x1:x2]
        try:
            face_r = cv2.resize(face_img, (100,100))
        except:
            continue

        face_r = img_to_array(face_r) / 255.0
        face_r = np.expand_dims(face_r, axis=0)
        (mask, no_mask) = model.predict(face_r, verbose=0)[0]

        label = "Mask" if mask>no_mask else "No Mask"
        color = (0,255,0) if label=="Mask" else (0,0,255)
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, f"{label} {max(mask,no_mask)*100:.1f}%", 
                    (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        # ── handle no-mask timing ────────────────────────────────────────
        if label=="No Mask":
            now = time.time()
            if current_id not in no_mask_timers:
                no_mask_timers[current_id] = now
            else:
                elapsed = now - no_mask_timers[current_id]
                # 5s warning
                if 5 < elapsed < 6:  
                    send_whatsapp(USER_WHATSAPP,
                        "⚠️ Please wear your mask. You’ve been maskless for over 5 seconds.")
                # 10s escalation
                if 10 < elapsed < 11:
                    send_whatsapp(MANAGER_WHATSAPP,
                        f"🚨 Alert: person at {current_id} has been maskless >10s.")
        else:
            # reset timer if they put mask on
            if current_id in no_mask_timers:
                del no_mask_timers[current_id]

    # clean up timers for faces gone
    for fid in list(no_mask_timers):
        if fid not in current:
            del no_mask_timers[fid]

    cv2.imshow("Mask Monitor", frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
