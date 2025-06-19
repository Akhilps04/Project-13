import cv2
import face_recognition
import pandas as pd
import os
import numpy as np
import pickle 
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from flask import Flask, render_template, Response, redirect, url_for, request, jsonify, session, make_response

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change to a secure key

# --------------- No-Cache Headers to Prevent Back Navigation ---------------
@app.after_request
def add_no_cache_headers(response):
    """
    Ensures that after logout, the browser doesn't serve a cached page
    when the user presses the back button.
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
# --------------------------------------------------------------------------

# Create attendance log CSV with full header (including Confidence) if it doesn't exist
if not os.path.exists("attendance_log.csv"):
    with open("attendance_log.csv", "w") as f:
        f.write("FullName,RollNumber,Department,Date,Time,Status,Confidence\n")


# ---------------------- Initialization ----------------------
student_details = pd.read_csv("student_details.csv")
COLLEGE_START_TIME = "09:30"  # Late if after this time

detection_logs = []

CHECK_CSV_FOR_DUPLICATES = False  # Set to True to skip already-marked students (after evaluation)

def add_log(message: str):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{now} {message}"
    detection_logs.append(log_entry)
    print(log_entry)

#................................# Load preprocessed face encodings from pickle (generated via save_encodings_pickle.py)..............................................................

with open("encodings.pkl", "rb") as f:
    student_encodings = pickle.load(f)

# ---------------------- Global Variables ----------------------
detection_active = False
unknown_count = 1
late_students_details = []
attendance_logged = set()  # Avoid duplicate records per day
last_detection_time = {}
COOL_DOWN_SECONDS = 30

cap = cv2.VideoCapture(0)

# ---------------------- Utility Functions ----------------------
def match_student(encoding, student_encodings, tolerance=0.6):
    best_match = None
    lowest_distance = float("inf")
    for first_name, stored_encoding in student_encodings.items():
        distance = face_recognition.face_distance([stored_encoding], encoding)[0]
        if distance < lowest_distance:
            lowest_distance = distance
            best_match = first_name
    if lowest_distance < tolerance:
        confidence = (1 - (lowest_distance / tolerance)) * 100
        return best_match, confidence
    return None, None

def is_late(current_time, college_start_time):
    current_time_obj = datetime.strptime(current_time, "%H:%M")
    start_time_obj = datetime.strptime(college_start_time, "%H:%M")
    return current_time_obj > start_time_obj

def save_unknown_face(face_image, count):
    unknown_folder = "UnknownFaces"
    os.makedirs(unknown_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(unknown_folder, f"unknown_{timestamp}_{count}.jpg")
    cv2.imwrite(filename, face_image)
    add_log(f"Unknown face saved as {filename}")

def record_attendance(full_name, roll_number, department, current_date, current_time, status,confidence):
    key = (full_name, current_date)
    if key not in attendance_logged:
        attendance_logged.add(key)
        record = f"{full_name},{roll_number},{department},{current_date},{current_time},{status},{confidence:.2f}\n"
        with open("attendance_log.csv", "a") as f:
            f.write(record)


def send_email(attendance_data):
    sender_email = "your_mail_id.com"
    receiver_email = "admin_mail_id.com"
    password = "app_password_mial"  # Replace with your password or env variable

    email_content = """
    <html>
    <body>
    <h2>Attendance Report</h2>
    <table border="1">
    <tr>
        <th>Full Name</th>
        <th>Roll Number</th>
        <th>Department</th>
        <th>Date</th>
        <th>Time</th>
        <th>Status</th>
    </tr>
    """
    for record in attendance_data:
        email_content += (
            f"<tr>"
            f"<td>{record['FullName']}</td>"
            f"<td>{record['RollNumber']}</td>"
            f"<td>{record['Department']}</td>"
            f"<td>{record['Date']}</td>"
            f"<td>{record['Time']}</td>"
            f"<td>{record['Status']}</td>"
            f"</tr>"
        )
    email_content += "</table></body></html>"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Attendance Report"
    msg.attach(MIMEText(email_content, 'html'))

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30)
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        add_log("Email sent successfully.")
    except Exception as e:
        add_log(f"Failed to send email: {e}")

# ---------------------- Video Streaming Generator ----------------------
def gen_frames():
    global unknown_count, detection_active, late_students_details
    while True:
        success, frame = cap.read()
        if not success:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_locations = [(top*4, right*4, bottom*4, left*4) for top, right, bottom, left in face_locations]

        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            name, roll_number, department = "Unknown", "Unknown", "Unknown"
            current_time = datetime.now().strftime("%H:%M")
            current_date = datetime.now().strftime("%Y-%m-%d")

            if detection_active:
                matched_first_name, confidence = match_student(face_encoding, student_encodings)
                if matched_first_name:
                    details = student_details[student_details['FullName'].str.contains(matched_first_name, case=False)]
                    if not details.empty:
                        full_name = details.iloc[0]['FullName']
                        department = details.iloc[0]['Department']
                        roll_number = details.iloc[0]['RollNumber']
                        name = full_name
                        status = "Late" if is_late(current_time, COLLEGE_START_TIME) else "On-Time"
                        
                        # Check if student is already marked today if flag is enabled
                        already_marked_today = False
                        if CHECK_CSV_FOR_DUPLICATES:
                            try:
                                existing_df = pd.read_csv("attendance_log.csv")
                                already_marked_today = not existing_df[
                                    (existing_df["FullName"] == full_name) &
                                    (existing_df["Date"] == current_date)
                                ].empty
                            except Exception as e:
                                add_log(f"Could not check existing CSV for duplicates: {e}")

                        now = datetime.now()
                        if not already_marked_today and (
                            (full_name not in last_detection_time) or
                            (now - last_detection_time[full_name] > timedelta(seconds=COOL_DOWN_SECONDS))
                        ):
                            last_detection_time[full_name] = now
                            record_attendance(full_name, roll_number, department, current_date, current_time, status,confidence)
                            late_students_details.append({
                                "FullName": full_name,
                                "RollNumber": roll_number,
                                "Department": department,
                                "Date": current_date,
                                "Time": current_time,
                                "Status": status,
                                "Confidence": f"{confidence:.2f}"
                            })
                            add_log(f"{status}: {full_name} - Roll No: {roll_number} - {department} (Confidence: {confidence:.2f}%)")
                    else:
                        add_log(f"Details for {matched_first_name} not found in CSV.")
                else:
                    save_unknown_face(frame[top:bottom, left:right], unknown_count)
                    unknown_count += 1

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ---------------------- Flask Routes ----------------------
@app.route('/')
def index():
    return render_template('index3.html', detection_active=detection_active)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_active
    if detection_active:
        add_log("Detection is already ON.")
    else:
        detection_active = True
        add_log("Detection is now ON.")
    return redirect(url_for('index'))

@app.route('/turn_off_detection', methods=['POST'])
def turn_off_detection():
    global detection_active
    if not detection_active:
        add_log("Detection is already OFF.")
    else:
        detection_active = False
        add_log("Detection is now OFF.")
    return redirect(url_for('index'))

@app.route('/send_email_report', methods=['POST'])
def send_email_report():
    global late_students_details
    add_log("Processing email report...")
    if late_students_details:
        send_email(late_students_details)
        late_students_details.clear()
    else:
        add_log("No attendance records to send. Email not sent.")
    return redirect(url_for('index'))

# ---------------------- Admin & Analytics Routes ----------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        # Simple check for demonstration:
        if username == "admin" and password == "admin123":
            session["logged_in"] = True
            return redirect(url_for("analytics"))
        else:
            error = "Invalid credentials. Please try again."
    return render_template("login.html", error=error)

@app.route('/logout')
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

@app.route('/analytics', methods=['GET', 'POST'])
def analytics():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        data = pd.read_csv("attendance_log.csv")
    except Exception as e:
        add_log(f"Error reading attendance log: {e}")
        data = pd.DataFrame(columns=["FullName","RollNumber","Department","Date","Time","Status"])

    # Filtering
    selected_date = None
    selected_department = None
    view_mode = 'summary'  # Default

    if request.method == 'POST':
        selected_date = request.form.get('date_filter')
        selected_department = request.form.get('department_filter')
        view_mode = request.form.get('view') or 'summary'

        if selected_date:
            data = data[data['Date'] == selected_date]
        if selected_department and selected_department != 'All':
            data = data[data['Department'] == selected_department]

    # Summary data (grouped by Date & Status)
    summary = []
    if view_mode == 'summary':
        summary = data.groupby(["Date", "Status"]).size().reset_index(name="Count").to_dict(orient="records")

    # Detailed records
    detailed_records = []
    if view_mode == 'detailed':
        detailed_records = data.to_dict(orient="records")

    # Present and Late counts
    present_count = data[data['Status'] == 'On-Time'].shape[0]
    absent_count = data[data['Status'] == 'Late'].shape[0]

    # Filters
    all_dates = sorted(data["Date"].unique().tolist())
    all_departments = sorted(data["Department"].dropna().unique().tolist())

    return render_template("analytics.html",
                           summary=summary,
                           all_dates=all_dates,
                           all_departments=all_departments,
                           selected_date=selected_date,
                           selected_department=selected_department,
                           view_mode=view_mode,
                           detailed_records=detailed_records,
                           present_count=present_count,
                           absent_count=absent_count)

@app.route('/download_attendance')
def download_attendance():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    status = request.args.get("status")  # "Late", "Present", or empty for all
    selected_date = request.args.get("date")
    selected_department = request.args.get("department")
    try:
        data = pd.read_csv("attendance_log.csv")
        if status:
            data = data[data["Status"] == status]
        if selected_date:
            data = data[data["Date"] == selected_date]
        if selected_department and selected_department != "All":
            data = data[data["Department"] == selected_department]        
        csv_data = data.to_csv(index=False)
        filename = f"attendance_{status or 'all'}_{selected_date or 'all'}.csv"
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        return f"Error downloading file: {e}"

@app.route('/logs')
def get_logs():
    return jsonify(detection_logs)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
