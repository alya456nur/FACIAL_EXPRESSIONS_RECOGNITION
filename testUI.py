import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

model = load_model('models/emotion_model1.keras')
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
img_size = 48

# Webcam detection
def detect_emotion_cam():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    window_name = 'Facial Expression Recognition - Webcam'
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)

        for (x, y, w, h) in faces:
            face_gray = grayscale[y:y+h, x:x+w]
            resized = cv2.resize(face_gray, (img_size, img_size))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, img_size, img_size, 1))

            prediction = model.predict(reshaped, verbose=0)[0]
            emotion_index = np.argmax(prediction)
            confidence = prediction[emotion_index] * 100
            label_text = f"{emotion_labels[emotion_index]} ({confidence:.1f}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)
        if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

# Image upload and display result in PyQt5
def predict_expression_image(parent=None):
    paths, _ = QFileDialog.getOpenFileNames(parent, "Select Image(s)", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"Could not read: {path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_size, img_size))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, img_size, img_size, 1))

        prediction = model.predict(reshaped, verbose=0)[0]
        emotion_index = np.argmax(prediction)
        confidence = prediction[emotion_index] * 100
        label = emotion_labels[emotion_index]

        # Show in a PyQt5 dialog
        dialog = QDialog(parent)
        dialog.setWindowTitle("Prediction Result")
        layout = QVBoxLayout(dialog)

        # Convert to QPixmap
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(180, 180, Qt.KeepAspectRatio)

        layout.addWidget(QLabel(os.path.basename(path)))
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)

        layout.addWidget(QLabel(f"Detected Emotion: {label}"))
        layout.addWidget(QLabel(f"Confidence: {confidence:.1f}%"))

        dialog.exec_()



# UI
#-------------------------------------------------------------------------------------------
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QSizePolicy,QFileDialog,QDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QImage
import sys

class ResponsiveWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Expression Recognition")
        self.background = QPixmap("images/backgroundImg.jpg")
        self.initUI()

    def initUI(self):
        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Container for widgets
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(20, 20, 20, 20)
        container_layout.setSpacing(15)

        # Title Label
        label = QLabel("Facial Expression Recognition")
        label.setAlignment(Qt.AlignHCenter)
        container_layout.addWidget(label)

        # Button setup
        buttons_text = ["Detect Emotion (Webcam)", "Predict from Image","Exit"]
        for text in buttons_text:
            button = QPushButton(text)
            button.setMinimumWidth(220)
            button.setStyleSheet("font-size: 14px; padding: 10px;")
            if text == "Exit":
                button.clicked.connect(self.close)
            elif text == "Detect Emotion (Webcam)":
                button.clicked.connect(detect_emotion_cam)
            elif text == "Predict from Image":
                button.clicked.connect(lambda _, p=self: predict_expression_image(parent=p))
            else:
                button.clicked.connect(lambda _, t=text: print(f"{t} clicked"))
            container_layout.addWidget(button, alignment=Qt.AlignHCenter)

        # Add container to main layout
        self.main_layout.addWidget(container, alignment=Qt.AlignCenter)

        # Dark theme styling
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(30, 136, 229, 0.85);
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: rgba(21, 101, 192, 0.9);
            }
            QLabel {
                color: white;
                font-size: 26px;
                font-weight: bold;
                background: transparent;
                border: 2px solid white;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
            }
            QDialog {
                background-color: #191970
;
            }
        """)

    def resizeEvent(self, event):
        # Resize background image with window
        scaled = self.background.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(scaled))
        self.setPalette(palette)
        super().resizeEvent(event)

# Run app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ResponsiveWindow()
    window.showMaximized()  # Make fullscreen
    sys.exit(app.exec_())
