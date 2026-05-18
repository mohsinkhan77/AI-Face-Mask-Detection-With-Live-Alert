# ==========================================
# IMPORT LIBRARIES
# ==========================================
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime

import time
import pyttsx3
import winsound
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = load_model("models/mask_detector.keras")



# ==========================================
# INITIALIZE TEXT TO SPEECH
# ==========================================

engine = pyttsx3.init()

engine.setProperty('rate', 150)

engine.setProperty('volume', 1.0)


# ==========================================
# ALERT CONTROL
# ==========================================

last_alert_time = 0

ALERT_DELAY = 7  # seconds

last_capture_time = 0

CAPTURE_DELAY = 30  # seconds

# ==========================================
# EMAIL CONFIGURATION
# ==========================================

SENDER_EMAIL = "mohsink0326@gmail.com"

SENDER_PASSWORD = "guygrsohqogqznlt"

RECEIVER_EMAIL = "mohsink0326@gmail.com"



# ==========================================
# LOAD HAAR CASCADE
# ==========================================

face_classifier = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

# ==========================================
# START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)


# ==========================================
# SEND EMAIL ALERT
# ==========================================

def send_email_alert(image_path):

    try:

        # Create email
        msg = MIMEMultipart()

        msg['Subject'] = "ALERT: No Mask Detected"

        msg['From'] = SENDER_EMAIL

        msg['To'] = RECEIVER_EMAIL

        # Email body
        body = """
        Warning!

        A person without a mask was detected.

        Please check the attached screenshot.
        """

        msg.attach(MIMEText(body, 'plain'))

        # Attach image
        with open(image_path, 'rb') as f:

            img_data = f.read()

            image = MIMEImage(img_data, name=image_path)

            msg.attach(image)

        # Connect Gmail server
        server = smtplib.SMTP("smtp.gmail.com", 587)

        server.starttls()

        # Login
        server.login(SENDER_EMAIL, SENDER_PASSWORD)

        # Send email
        server.send_message(msg)

        # Close server
        server.quit()

        print("[INFO] Email alert sent successfully!")

    except Exception as e:

        print("[ERROR] Email failed:", e)



# ==========================================
# LOOP THROUGH VIDEO FRAMES
# ==========================================

while True:

    # Read webcam frame
    ret, frame = cap.read()

    # Break if webcam fails
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # ==========================================
    # LOOP THROUGH DETECTED FACES
    # ==========================================

    for (x, y, w, h) in faces:

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Convert BGR → RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Resize face
        face = cv2.resize(face, (224, 224))

        # Convert to array
        face = img_to_array(face)

        # Normalize image
        face = preprocess_input(face)

        # Expand dimensions
        face = np.expand_dims(face, axis=0)

        # ==========================================
        # PREDICT MASK / NO MASK
        # ==========================================

        prediction = model.predict(face, verbose=0)

        mask, withoutMask = prediction[0]

        # ==========================================
        # MASK DETECTED
        # ==========================================

        if mask > withoutMask:

            label = "Mask"

            color = (0, 255, 0)

            confidence = mask * 100

        # ==========================================
        # NO MASK DETECTED
        # ==========================================

        else:

            label = "No Mask"

            color = (0, 0, 255)

            confidence = withoutMask * 100

        # ==========================================
        # SMART ALERT SYSTEM
        # ==========================================

        # Get current time
            current_time = time.time()

        # Check alert cooldown
            if current_time - last_alert_time > ALERT_DELAY:

            # Speak warning
                engine.say("Please wear your mask")

                engine.runAndWait()

            # Update last alert time
                last_alert_time = current_time

                # ==========================================
            # SAVE SCREENSHOT
            # ==========================================

            if current_time - last_capture_time > CAPTURE_DELAY:

        # Generate timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Image filename
                filename = f"captures/no_mask_{timestamp}.jpg"

        # Save image
                cv2.imwrite(filename, frame)

                send_email_alert(filename)

                print(f"[INFO] Screenshot saved: {filename}")

        # Update capture timer
                last_capture_time = current_time

               
        # Create label text
        text = f"{label}: {confidence:.2f}%"

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Put text
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # ==========================================
    # SHOW VIDEO
    # ==========================================

    cv2.imshow("Face Mask Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# RELEASE RESOURCES
# ==========================================

cap.release()

cv2.destroyAllWindows()