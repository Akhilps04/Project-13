import os
import pickle
import numpy as np
import face_recognition
import cv2

# ---------------------- Image Augmentation Functions ----------------------
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def adjust_brightness(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
    hsv[:, :, 2] *= factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype("uint8")
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augment_image(image):
    augmented = [image]  # Original
    augmented.append(rotate_image(image, 10))
    augmented.append(rotate_image(image, -10))
    augmented.append(adjust_brightness(image, 1.2))
    augmented.append(adjust_brightness(image, 0.8))
    return augmented

# ---------------------- Save Encodings ----------------------
def save_encodings_pickle(database_path="StudentDatabase", output_path="encodings.pkl"):
    student_encodings = {}
    for filename in os.listdir(database_path):
        if filename.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(database_path, filename)
            img = face_recognition.load_image_file(img_path)
            aug_images = augment_image(img)
            encodings = []
            for aug_img in aug_images:
                enc = face_recognition.face_encodings(aug_img)
                if enc:
                    encodings.append(enc[0])
            if encodings:
                avg_encoding = np.mean(encodings, axis=0)
                name = os.path.splitext(filename)[0]
                student_encodings[name] = avg_encoding
                print(f"[INFO] Encoding saved for {name}")
            else:
                print(f"[WARNING] No face found in {filename}")

    with open(output_path, "wb") as f:
        pickle.dump(student_encodings, f)
    print(f"\n✅ All encodings saved to {output_path}")

# ---------------------- Run It ----------------------
if __name__ == "__main__":
    save_encodings_pickle()
