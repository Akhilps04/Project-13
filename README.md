# Chrono-Guard: AI-Driven Campus Sentinel
"An AI-powered real-time attendance monitoring system using facial recognition and time-based status detection to automate and streamline attendance management in educational institutions."

## Features
🎥 Real-time face recognition using webcam input (via face_recognition + OpenCV)

📍 Time-based categorization of students as On-Time or Late

📄 Automatic attendance logging with name, roll number, department, timestamp, and confidence

📊 Visual analytics dashboard with daily filtering by date and department

📤 Export attendance records (on-time, late, or all) as CSV

📧 Email reporting of late attendance (admin-controlled)

🔒 Admin login with session handling and secure logout

# ⚙️ Requirements
Make sure the following dependencies are installed:

"pip install opencv-python face_recognition flask pandas numpy"
Additional tools:
Python 3.8+
Webcam access

A GPU is optional but helps with performance.
## 🔐 Admin Login
Username: admin
Password: admin123

You can change this in app.py for better security.
## 📬 Email Reporting (Optional)
To enable email reporting, configure your Gmail credentials inside send_email() function in app.py. Make sure to allow access for less secure apps or use an app password.

## 🧠 Future Enhancements
Multi-camera support for large campuses

Integration with LMS (e.g., Moodle)

Absence detection by cross-checking unrecognized faces

Mobile app interface

Cloud-based deployment

# Acknowledgments
The dataset was sourced from Kaggle.
Thanks to my mentors and colleagues for their guidance.

## 📄 License
This project is for academic and demonstration purposes only. Please credit the authors if reused or modified.
