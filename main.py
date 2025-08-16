import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import tempfile
import glob
import os

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Helper functions
def get_landmarks(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        h, w, _ = image.shape
        return [(int(float(pt.x) * w), int(float(pt.y) * h)) for pt in results.multi_face_landmarks[0].landmark]
    return None

def add_romantic_theme(img):
    # No pink overlay, just return the image
    return img

st.set_page_config(page_title="Past Life Lover Past Life Predictor", layout="centered")
# Retro style CSS
st.markdown("""
    <style>
    body {
        background: #f5ecd7;
        font-family: 'Georgia', serif;
    }
    .retro-title {
        font-family: 'Georgia', serif;
        font-size: 3em;
        color: #5b3a29;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 0px;
        letter-spacing: 2px;
        text-shadow: 2px 2px 0 #e2c9a0, 4px 4px 0 #c2a77d;
    }
    .retro-desc {
        font-family: 'Georgia', serif;
        color: #7c5c3b;
        text-align: center;
        font-size: 1.3em;
        margin-bottom: 30px;
        margin-top: -10px;
        letter-spacing: 1px;
    }
    .retro-box {
        background: #e2c9a0;
        border: 3px solid #5b3a29;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 0 20px #c2a77d;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='retro-title'>MAGADHEERA</div>", unsafe_allow_html=True)
st.markdown("<div class='retro-desc'>past life predictor</div>", unsafe_allow_html=True)

# Only webcam input
st.markdown("<div class='retro-box'>", unsafe_allow_html=True)
img = None
img_bytes = st.camera_input("Capture your photo from webcam")
if img_bytes:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(img_bytes.getvalue())
        tmp_path = tmp_file.name
    img = cv2.imread(tmp_path)
    img = cv2.resize(img, (640, 480))
st.markdown("</div>", unsafe_allow_html=True)

if img is not None:
    # Find all reference images in folder
    import random
    ref_folder = "./"
    ref_imgs = sorted(glob.glob(os.path.join(ref_folder, '*.jpg')) + glob.glob(os.path.join(ref_folder, '*.png')))
    if not ref_imgs:
        st.error(f"No reference images found in {ref_folder}")
    else:
        st.write(f"Found {len(ref_imgs)} reference images.")
        # Pick a random reference image
        ref_img_path = random.choice(ref_imgs)
        ref_img = cv2.imread(ref_img_path)
        if ref_img is None:
            st.error(f"Could not read reference image: {ref_img_path}")
        else:
            ref_img = cv2.resize(ref_img, (640, 480))
            # Detect faces in both images
            captured_landmarks = get_landmarks(img)
            ref_landmarks = get_landmarks(ref_img)
            img_copy = img.copy()
            if captured_landmarks and ref_landmarks:
                # Get bounding box for captured face
                xs1 = [pt[0] for pt in captured_landmarks]
                ys1 = [pt[1] for pt in captured_landmarks]
                min_x1, max_x1 = min(xs1), max(xs1)
                min_y1, max_y1 = min(ys1), max(ys1)
                pad_x1 = int((max_x1 - min_x1) * 0.15)
                pad_y1 = int((max_y1 - min_y1) * 0.25)
                min_x1 = max(min_x1 - pad_x1, 0)
                max_x1 = min(max_x1 + pad_x1, img_copy.shape[1]-1)
                min_y1 = max(min_y1 - pad_y1, 0)
                max_y1 = min(max_y1 + pad_y1, img_copy.shape[0]-1)
                face_w1 = max_x1 - min_x1
                face_h1 = max_y1 - min_y1
                # Get bounding box for reference face
                xs2 = [pt[0] for pt in ref_landmarks]
                ys2 = [pt[1] for pt in ref_landmarks]
                min_x2, max_x2 = min(xs2), max(xs2)
                min_y2, max_y2 = min(ys2), max(ys2)
                pad_x2 = int((max_x2 - min_x2) * 0.15)
                pad_y2 = int((max_y2 - min_y2) * 0.25)
                min_x2 = max(min_x2 - pad_x2, 0)
                max_x2 = min(max_x2 + pad_x2, ref_img.shape[1]-1)
                min_y2 = max(min_y2 - pad_y2, 0)
                max_y2 = min(max_y2 + pad_y2, ref_img.shape[0]-1)
                face_w2 = max_x2 - min_x2
                face_h2 = max_y2 - min_y2
                # Crop reference face
                ref_face_crop = ref_img[min_y2:max_y2, min_x2:max_x2]
                # Resize reference face to fit captured face
                ref_face_resized = cv2.resize(ref_face_crop, (face_w1, face_h1))
                ref_light = cv2.convertScaleAbs(ref_face_resized, alpha=1.2, beta=100)
                alpha_overlay = 0.7
                roi = img_copy[min_y1:max_y1, min_x1:max_x1]
                blended = cv2.addWeighted(ref_light, alpha_overlay, roi, 1 - alpha_overlay, 0)
                img_copy[min_y1:max_y1, min_x1:max_x1] = blended
                cv2.rectangle(img_copy, (min_x1, min_y1), (max_x1, max_y1), (0,0,255), 2)
            else:
                # Fallback: overlay in top-left
                overlay_size = (160, 120)
                ref_small = cv2.resize(ref_img, overlay_size)
                ref_light = cv2.convertScaleAbs(ref_small, alpha=1.2, beta=100)
                alpha_overlay = 0.7
                roi = img_copy[0:overlay_size[1], 0:overlay_size[0]]
                blended = cv2.addWeighted(ref_light, alpha_overlay, roi, 1 - alpha_overlay, 0)
                img_copy[0:overlay_size[1], 0:overlay_size[0]] = blended
                cv2.rectangle(img_copy, (0,0), (overlay_size[0]-1, overlay_size[1]-1), (0,0,255), 2)
            themed_img = add_romantic_theme(img_copy)
            st.markdown("<h3 style='text-align: center;'>past life</h3>", unsafe_allow_html=True)
            # Crop output to only the overlayed face region
            if captured_landmarks and ref_landmarks:
                xs1 = [pt[0] for pt in captured_landmarks]
                ys1 = [pt[1] for pt in captured_landmarks]
                min_x1, max_x1 = min(xs1), max(xs1)
                min_y1, max_y1 = min(ys1), max(ys1)
                pad_x1 = int((max_x1 - min_x1) * 0.15)
                pad_y1 = int((max_y1 - min_y1) * 0.25)
                min_x1 = max(min_x1 - pad_x1, 0)
                max_x1 = min(max_x1 + pad_x1, themed_img.shape[1]-1)
                min_y1 = max(min_y1 - pad_y1, 0)
                max_y1 = min(max_y1 + pad_y1, themed_img.shape[0]-1)
                crop_img = themed_img[min_y1:max_y1, min_x1:max_x1]
            else:
                overlay_size = (160, 120)
                crop_img = themed_img[0:overlay_size[1], 0:overlay_size[0]]
            st.image(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

