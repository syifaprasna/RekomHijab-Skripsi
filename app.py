import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import base64
import urllib.request
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input


# 1. config halaman dan css
st.set_page_config(
    page_title="Portal Deteksi Bentuk Wajah",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Blok utama css semua halaman
st.markdown("""
<style>
html {
    scroll-behavior: smooth;
}
[data-testid="stHeader"] {
    display: none;
}

#deteksi, #tata-cara, #video, #faq {
    scroll-margin-top: 120px;
}

.custom-navbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background-color: #ffffff;
    padding: 12px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.04);
    z-index: 99999;
    border-bottom: 2px solid #FFF0F2;
    box-sizing: border-box;
}

.nav-logo {
    font-size: 20px;
    font-weight: bold;
    color: #F13E93 !important;
    text-decoration: none !important;
    border: none !important;
    outline: none !important;
}

.nav-logo:hover, .nav-logo:focus, .nav-logo:active {
    text-decoration: none !important;
    color: #F13E93 !important;
}

.nav-links {
    display: flex;
    gap: 12px;
    align-items: center;
}

.nav-links a {
    color: #827397;
    text-decoration: none;
    font-weight: 600;
    font-size: 15px;
    padding: 8px 22px;
    border-radius: 50px;
    transition: all 0.3s ease;
}

.nav-links a:hover, .nav-links a:focus, .nav-links a:active {
    background: linear-gradient(135deg, #FF7EB3 0%, #F13E93 100%);
    color: #FFFFFF !important;
    box-shadow: 0px 4px 12px rgba(241, 62, 147, 0.3);
}

/* MEDIA QUERY HP / TABLET */
@media screen and (max-width: 768px) {
    .custom-navbar { padding: 10px 20px; }
    .desktop-links { display: none !important; }
    .mobile-menu-container { display: block !important; }
}

@media screen and (min-width: 769px) {
    .mobile-menu-container { display: none !important; }
}

.mobile-menu-container summary {
    list-style: none;
    font-size: 26px;
    color: #F13E93;
    cursor: pointer;
    user-select: none;
}
.mobile-menu-container summary::-webkit-details-marker { display: none; }

.mobile-dropdown {
    position: absolute;
    top: 50px;
    left: 0;
    width: 100%;
    background-color: #ffffff;
    padding: 15px 0;
    box-shadow: 0px 10px 15px rgba(0,0,0,0.08);
    border-bottom: 2px solid #FFF0F2;
    display: flex;
    flex-direction: column;
    gap: 12px;
    text-align: center;
}
.mobile-dropdown a {
    color: #827397;
    text-decoration: none;
    font-weight: 600;
    font-size: 15px;
    padding: 10px 20px;
    margin: 0 20px;
    border-radius: 50px;
    transition: all 0.3s ease;
}
.mobile-dropdown a:hover, .mobile-dropdown a:focus, .mobile-dropdown a:active {
    background: linear-gradient(135deg, #FF7EB3 0%, #F13E93 100%);
    color: #FFFFFF !important;
}


.main-content {
    margin-top: 25px;
}

.section-title {
    text-align: center;
    margin-top: 0px;
    margin-bottom: 5px;
    color: #4A3E56;
}

.section-subtitle {
    text-align: center;
    color: #A093AE;
    margin-bottom: 40px;
    font-size: 14px;
}

.centered-sub {
    text-align: center;
    color: #4A3E56;
    margin-top: 25px;
    margin-bottom: 20px;
}

[data-testid="stImage"] img {
    max-height: 300px !important;
    object-fit: contain;
    border-radius: 8px;
    display: block;
    margin-left: auto;
    margin-right: auto;
}


.dashboard-card {
    background: #FFFFFF;
    border: 1.5px solid #FFE0EC;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0px 8px 24px rgba(241, 62, 147, 0.08);
    margin: 20px 0;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.dashboard-card:hover {
    box-shadow: 0px 12px 30px rgba(241, 62, 147, 0.12);
}

.hero-box {
    background: linear-gradient(135deg, #FFF0F5 0%, #FFE4EF 100%);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    border: 1px solid #FFD0DF;
    margin-bottom: 20px;
}
.hero-badge {
    display: inline-block;
    background: #F13E93;
    color: #FFFFFF;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 50px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.hero-title {
    font-size: 22px;
    font-weight: 800;
    color: #4A3E56;
    margin: 0;
}
.hero-highlight { color: #F13E93; }

.prob-card-item {
    background: #FAF8FC;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #F3EDF5;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.prob-card-item:hover {
    transform: translateY(-2px);
    border-color: #FFC0DB;
}
.prob-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    font-weight: 700;
    color: #4A3E56;
    margin-bottom: 8px;
}
.custom-bar-bg {
    background-color: #EFEAF2;
    height: 8px;
    border-radius: 10px;
    overflow: hidden;
}
.custom-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #FF7EB3 0%, #F13E93 100%);
    border-radius: 10px;
    transition: width 0.8s ease-in-out;
}

.gradcam-footer-note {
    background-color: #FFF9FC;
    border-left: 4px solid #F13E93;
    padding: 12px 16px;
    font-size: 13px;
    color: #6C5F78;
    border-radius: 0 10px 10px 0;
    margin-top: 20px;
    line-height: 1.5;
}

.mua-box {
    background: #FFFFFF;
    border: 1.5px solid #FFE0EC;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0px 4px 15px rgba(241, 62, 147, 0.04);
    margin-top: 20px;
}
.mua-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #F13E93;
    font-weight: 700;
    font-size: 13px;
    background: #FFF0F5;
    padding: 6px 14px;
    border-radius: 30px;
    margin-bottom: 12px;
}
.mua-text {
    color: #4A3E56;
    font-size: 14px;
    line-height: 1.6;
    margin: 0;
    text-align: justify;
}

[data-testid="stCameraInput"] video {
    transform: scaleX(-1) !important;
}


.faq-section-wrapper {
    background-color: #FFF0F5;
    padding: 40px 30px;
    margin: 40px auto 10px auto;
    max-width: 900px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(241, 62, 147, 0.06);
}
.faq-container-inner {
    max-width: 800px;
    margin: 0 auto;
}
.faq-item {
    background-color: #FFFFFF;
    border: 1.5px solid #FFE0EC;
    border-radius: 14px;
    margin-bottom: 12px;
    overflow: hidden;
    box-shadow: 0px 4px 10px rgba(241, 62, 147, 0.03);
    transition: all 0.3s ease;
}
.faq-item summary {
    padding: 16px 20px;
    font-weight: 700;
    color: #4A3E56;
    cursor: pointer;
    font-size: 15px;
    list-style: none;
    display: flex;
    justify-content: space-between;
    align-items: center;
    user-select: none;
}
.faq-item summary::-webkit-details-marker { display: none; }
.faq-item summary::after {
    content: "❯";
    font-size: 12px;
    color: #F13E93;
    transition: transform 0.3s ease;
}
.faq-item[open] summary::after { transform: rotate(90deg); }
.faq-answer {
    padding: 0 20px 18px 20px;
    color: #6C5F78;
    font-size: 14px;
    line-height: 1.6;
    border-top: 1px solid #FFF0F2;
    margin-top: 5px;
    padding-top: 12px;
}

.cta-banner-box {
    text-align: center;
    max-width: 700px;
    margin: 60px auto 30px auto;
    padding: 0 20px;
}
.cta-title {
    font-size: 22px;
    font-weight: 800;
    color: #4A3E56;
    margin-bottom: 8px;
}
.cta-subtitle {
    font-size: 14px;
    color: #827397;
    margin-bottom: 25px;
}

.btn-coba-sekarang {
    display: inline-block;
    background: linear-gradient(135deg, #FF7EB3 0%, #F13E93 100%);
    color: #FFFFFF !important;
    font-weight: 700;
    font-size: 15px;
    padding: 14px 38px;
    border-radius: 50px;
    text-decoration: none !important;
    box-shadow: 0px 6px 18px rgba(241, 62, 147, 0.35);
    transition: all 0.3s ease;
}
.btn-coba-sekarang:hover {
    transform: translateY(-3px);
    box-shadow: 0px 10px 25px rgba(241, 62, 147, 0.5);
    color: #FFFFFF !important;
}

.faq-footer {
    text-align: center;
    padding-top: 25px;
    margin-top: 50px;
    border-top: 1px solid #FFF0F2;
    color: #827397;
}
.faq-footer h5 {
    color: #F13E93;
    font-weight: bold;
    margin: 0 0 5px 0;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# HTML STICKY NAVBAR
navbar_html = """
<div class="custom-navbar">
<a class="nav-logo" href="#">RekomHijab</a>
<div class="nav-links desktop-links">
<a href="#deteksi">Deteksi</a>
<a href="#tata-cara">Tata Cara</a>
<a href="#video">Video</a>
<a href="#faq">FAQ</a>
</div>
<div class="mobile-menu-container">
<details>
<summary>☰</summary>
<div class="mobile-dropdown">
<a href="#deteksi">Deteksi</a>
<a href="#tata-cara">Tata Cara</a>
<a href="#video">Video</a>
<a href="#faq">FAQ</a>
</div>
</details>
</div>
</div>
"""
st.markdown(navbar_html, unsafe_allow_html=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)


# 2. Load model dan main logic
MODEL_PATH = "best_resnet50v2_face.h5"
LABELS = ['Heart', 'Oval', 'Round', 'Square']

@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_my_model()

@st.cache_resource
def get_yunet_model():
    model_file = "face_detection_yunet_2023mar.onnx"
    if not os.path.exists(model_file):
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        try:
            urllib.request.urlretrieve(url, model_file)
        except Exception as e:
            print(f"Gagal mengunduh YuNet model: {e}")
            return None
    return model_file

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="resnet50v2", sub_layer_name="post_relu", pred_index=None):
    base_model = model.get_layer(last_conv_layer_name)
    inner_grad_model = tf.keras.models.Model(inputs=[base_model.inputs], outputs=[base_model.get_layer(sub_layer_name).output])
    with tf.GradientTape() as tape:
        conv_outputs = inner_grad_model(img_array)
        x = conv_outputs
        for layer in model.layers[1:]:
            x = layer(x)
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap_max = tf.reduce_max(heatmap)
    if heatmap_max == 0: heatmap_max = 1e-5
    return (heatmap / heatmap_max).numpy()

def crop_face_keep_ratio(img_rgb):
    if img_rgb is None:
        return None

    try:
        img_clean = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        h, w, _ = img_clean.shape
        img_bgr = cv2.cvtColor(img_clean, cv2.COLOR_RGB2BGR)

        # 1. METODE UTAMA: YuNet DNN Face Detector (Sangat Akurat, Tanpa Salah Crop Pohon)
        yunet_onnx = get_yunet_model()
        if yunet_onnx and os.path.exists(yunet_onnx) and hasattr(cv2, 'FaceDetectorYN'):
            detector = cv2.FaceDetectorYN.create(
                model=yunet_onnx,
                config="",
                input_size=(w, h),
                score_threshold=0.5,
                nms_threshold=0.3,
                top_k=5000
            )
            _, faces = detector.detect(img_bgr)

            if faces is not None and len(faces) > 0:
                # Pilih wajah dengan skor confidence/luas terbesar
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                box = faces[0][:4].astype(int)
                x, y, w_box, h_box = box

                # Pengecekan rasio aspek wajah manusia (lebar / tinggi)
                aspect_ratio = float(w_box) / max(1, h_box)
                if 0.5 <= aspect_ratio <= 1.5:
                    margin = int(0.2 * max(w_box, h_box))
                    y1 = max(0, y - margin)
                    y2 = min(h, y + h_box + margin)
                    x1 = max(0, x - margin)
                    x2 = min(w, x + w_box + margin)
                    return img_rgb[y1:y2, x1:x2]

    except Exception as e:
        print(f"Error YuNet: {e}")

    # 2. METODE FALLBACK: Haar Cascade dengan Parameter Ketat
    try:
        xml_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(xml_path)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        # minNeighbors=6 & minSize=(60, 60) mencegah dedaunan/pohon ter-crop
        faces = face_cascade.detectMultiScale(
            gray_eq, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(60, 60)
        )

        if len(faces) > 0:
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            x, y, w_box, h_box = faces[0]

            aspect_ratio = float(w_box) / max(1, h_box)
            if 0.6 <= aspect_ratio <= 1.4:
                margin = int(0.2 * max(w_box, h_box))
                y1 = max(0, y - margin)
                y2 = min(img_rgb.shape[0], y + h_box + margin)
                x1 = max(0, x - margin)
                x2 = min(img_rgb.shape[1], x + w_box + margin)
                return img_rgb[y1:y2, x1:x2]

    except Exception as e:
        print(f"Error Haar Fallback: {e}")

    # JIKA BUKAN WAJAH MANUSIA (Pohon, Bola, Barang): RETURN NONE!
    return None

def get_hijab_recommendation(shape_label):
    recommendations = {
        'Heart': {
            'penjelasan': "Bentuk wajah Heart atau hati (sering disebut bentuk 'daun sirih') memiliki struktur dagu V-shape yang sangat proporsional. Karakteristik wajah ini sangat fleksibel dan cocok menggunakan model hijab apapun.",
            'gaya': [
                {"foto": "asetRekomhijab/GenerateHijab/Haert/heart1.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Haert/heart2.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Haert/heart3.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Haert/heart4.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Haert/heart5.png"}
            ]
        },
        'Oval': {
            'penjelasan': "Bentuk wajah Oval merupakan bentuk wajah yang paling ideal dan proporsional. Pemilik wajah oval sangat beruntung karena cocok memakai berbagai macam gaya hijab, mulai dari pashmina lilit, gaya segi empat konvensional, hingga model bergo instan.",
            'gaya': [
                {"foto": "asetRekomhijab/GenerateHijab/Oval/oval1.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Oval/oval2.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Oval/oval3.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Oval/oval4.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Oval/oval5.png"}
            ]
        },
        'Round': {
            'penjelasan': "Bentuk wajah Round (bulat), disarankan memilih gaya pemasangan hijab yang agak dimajukan ke arah depan pipi. Teknik ini bertujuan untuk menutupi sebagian area pipi sehingga memberikan ilusi wajah yang tampak lebih tirus, panjang, dan simetris..",
            'gaya': [
                {"foto": "asetRekomhijab/GenerateHijab/Round/round1.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Round/round2.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Round/round3.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Round/round4.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Round/round5.png"}
            ]
        },
        'Square': {
            'penjelasan': "bentuk wajah Square (persegi) disarankan untuk melipat kedua sisi hijab tepat di bawah pipi sehingga kain dapat menutupi sebagian area pipi dan rahang. Teknik framing ini efektif untuk menyamarkan garis rahang yang tegas agar tampilan wajah terlihat lebih lembut dan membulat.",
            'gaya': [
                {"foto": "asetRekomhijab/GenerateHijab/Square/square1.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Square/square2.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Square/square3.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Square/square4.png"},
                {"foto": "asetRekomhijab/GenerateHijab/Square/square5.png"}
            ]
        }
    }
    return recommendations.get(shape_label)


# SECTION 1: DETEKSI
st.markdown('<div id="deteksi"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Temukan model hijab Anda</h2>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Ambil gambar secara langsung atau unggah file foto wajah dan temukan model hijab sesuai dengan bentuk wajah Anda!</p>', unsafe_allow_html=True)

if model is None:
    st.error(f"File model `{MODEL_PATH}` tidak ditemukan. Harap pastikan ditaruh di folder yang sama.")
else:
    _, input_center_block, _ = st.columns([0.15, 0.7, 0.15])
    
    source_img = None
    
    with input_center_block:
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            camera_file = st.camera_input("Kamera")
        with input_col2:
            uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

        if camera_file is not None:
            source_img = Image.open(camera_file)
        elif uploaded_file is not None:
            source_img = Image.open(uploaded_file)
        else:
            source_img = None

    if source_img is not None:
        img_rgb = np.array(source_img.convert("RGB"))
        
        with st.spinner("Memproses..."):
            face_ready = crop_face_keep_ratio(img_rgb)
            
        _, output_center_block, _ = st.columns([0.15, 0.7, 0.15])
        
        with output_center_block:
            if face_ready is None:
                st.markdown("<br>", unsafe_allow_html=True)
                st.error("Wajah tidak terdeteksi!")
            else:
                img_resized = cv2.resize(face_ready, (299, 299))
                img_array = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img_resized), axis=0)
                processed_img = preprocess_input(img_array)
                
                preds = model.predict(processed_img)
                pred_idx = tf.argmax(preds[0]).numpy()
                confidence = preds[0][pred_idx] * 100
                detected_shape = LABELS[pred_idx]
                
                heatmap = make_gradcam_heatmap(processed_img, model)
                
                h_img, w_img = face_ready.shape[:2]
                heatmap_resized = cv2.resize(np.uint8(255 * heatmap), (w_img, h_img))
                
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h3 class='centered-sub'>Hasil Deteksi</h3>", unsafe_allow_html=True)
             
                res_img_col1, res_img_col2 = st.columns(2)
                
                with res_img_col1:
                    st.image(face_ready, caption="Hasil Crop Wajah", use_container_width=True)
                    
                with res_img_col2:
                    fig, ax = plt.subplots(figsize=(4, 4 * (h_img / w_img)))

                    ax.imshow(face_ready)
                    
                    ax.imshow(
                        heatmap_resized, 
                        cmap='jet', 
                        alpha=0.35, 
                        extent=[0, w_img, h_img, 0], 
                        aspect='auto'
                    )
                    
                    ax.axis('off')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    st.pyplot(fig, use_container_width=True)

                # Dashboard hasil deteksi
                st.markdown(f"""
                <div class="dashboard-card">
                    <div class="hero-box">
                        <span class="hero-badge">Hasil Deteksi</span>
                        <h3 class="hero-title">Bentuk Wajah: <span class="hero-highlight">{detected_shape}</span> ({confidence:.2f}%)</h3>
                    </div>
                """, unsafe_allow_html=True)

                # Bar face shape
                prob_cols = st.columns(4)
                for i, label in enumerate(LABELS):
                    prob_val = preds[0][i] * 100
                    with prob_cols[i]:
                        st.markdown(f"""
                        <div class="prob-card-item">
                            <div class="prob-label-row">
                                <span>{label}</span>
                                <span style="color:#F13E93;">{prob_val:.1f}%</span>
                            </div>
                            <div class="custom-bar-bg">
                                <div class="custom-bar-fill" style="width: {prob_val}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Catatan Grad-CAM
                st.markdown(f"""
                    <div class="gradcam-footer-note">
                        <strong>Grad-CAM:</strong> Fokus utama AI berada pada sorotan bercak <span style="color:red; font-weight:bold;">merah</span> untuk menyimpulkan bentuk wajah <strong>{detected_shape}</strong>.
                    </div>
                </div> <!-- TUTUP DASHBOARD CARD -->
                """, unsafe_allow_html=True)

                # Rekomendasi model hijab (card MUA)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h3 class='centered-sub'>Rekomendasi Model Hijab</h3>", unsafe_allow_html=True)
                
                data_hijab = get_hijab_recommendation(detected_shape)
                
                st.markdown(f"""
                <div class="mua-box">
                    <div class="mua-badge">
                        Penjelasan Hasil Konsultasi MUA
                    </div>
                    <p class="mua-text">
                        {data_hijab['penjelasan']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                def get_image_base64(path):
                    with open(path, "rb") as image_file:
                        return base64.b64encode(image_file.read()).decode()

                html_items = ""
                for index, item in enumerate(data_hijab['gaya']):
                    if os.path.exists(item['foto']):
                        try:
                            img_base64 = get_image_base64(item['foto'])
                            content_box = f'<img src="data:image/jpeg;base64,{img_base64}" style="width:100%; height:200px; object-fit:contain; border-radius:8px; margin-bottom:8px;">'
                        except Exception as e:
                            content_box = f'<div style="width:100%; height:280px; background-color:#FFFDEB; display:flex; align-items:center; justify-content:center; color:red; font-size:11px;">Error</div>'
                    else:
                        content_box = f"""
                        <div style="width:100%; height:200px; background-color:#FFFDEB; border:2px dashed #E6C845; border-radius:8px; display:flex; align-items:center; justify-content:center; text-align:center; padding:10px; color:#B0941A; font-size:11px; font-weight:bold; box-sizing:border-box;">
                             [Gambar {index+1}]
                        </div>
                        """
                        
                    html_items += f"""
                    <div style="flex: 0 0 220px; scroll-snap-align: start; background-color: #FFFFFF; border: 1px solid #EFEAEF; border-radius: 12px; padding: 12px; box-shadow: 0px 4px 6px rgba(0,0,0,0.02); text-align: center; font-family: sans-serif; box-sizing: border-box;">
                        {content_box}
                    </div>
                    """
                
                full_carousel_html = f"""
                <div style="display: flex; overflow-x: auto; gap: 15px; padding: 10px; scroll-snap-type: x mandatory; -webkit-overflow-scrolling: touch; background-color: transparent;">
                    {html_items}
                </div>
                <style>
                    ::-webkit-scrollbar {{ height: 6px; }}
                    ::-webkit-scrollbar-track {{ background: #f1f1f1; border-radius: 10px; }}
                    ::-webkit-scrollbar-thumb {{ background: #FE81D4; border-radius: 10px; }}
                </style>
                """
                
                st.components.v1.html(full_carousel_html, height=360, scrolling=False)

st.markdown("---")

# SECTION 2: TATA CARA
st.markdown('<div id="tata-cara"></div>', unsafe_allow_html=True)
st.markdown('<h3 class="section-title">TATA CARA PENGGUNAAN</h3>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Ikuti panduan berikut agar hasil deteksi bentuk wajah menjadi maksimal</p>', unsafe_allow_html=True)

_, tata_cara_block, _ = st.columns([0.15, 0.7, 0.15])

with tata_cara_block:
    st.markdown("<h4 style='color: #4A3E56; margin-bottom: 15px;'>Kamera:</h4>", unsafe_allow_html=True)
    
    cam_col1, cam_col2, cam_col3 = st.columns(3)
    with cam_col1:
        st.markdown("""
            <div style="background-color: #FFF2F6; border-left: 4px solid #FE81D4; padding: 15px; border-radius: 8px; min-height: 140px;">
                <strong style="color: #4A3E56;">1. Izinkan Akses</strong><br><br>Setujui permintaan browser untuk mengakses kamera depan perangkat Anda.
            </div>
        """, unsafe_allow_html=True)
    with cam_col2:
        st.markdown("""
            <div style="background-color: #FFF2F6; border-left: 4px solid #FE81D4; padding: 15px; border-radius: 8px; min-height: 140px;">
                <strong style="color: #4A3E56;">2. Atur Posisi</strong><br><br>Posisikan wajah tegak lurus menatap kamera (simetris) dengan pencahayaan ruangan yang terang.
            </div>
        """, unsafe_allow_html=True)
    with cam_col3:
        st.markdown("""
            <div style="background-color: #FFF2F6; border-left: 4px solid #FE81D4; padding: 15px; border-radius: 8px; min-height: 140px;">
                <strong style="color: #4A3E56;">3. Ambil Gambar</strong><br><br>Tekan tombol 'Take Photo' di bawah kotak kamera untuk memulai deteksi otomatis.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("<h4 style='color: #4A3E56; margin-bottom: 15px;'>Unggah Gambar:</h4>", unsafe_allow_html=True)
    
    up_col1, up_col2, up_col3 = st.columns(3)
    with up_col1:
        st.markdown("""
            <div style="background-color: #FAF8FF; border-left: 4px solid #8A72A4; padding: 15px; border-radius: 8px; min-height: 140px;">
                <strong style="color: #4A3E56;">1. Pilih File</strong><br><br>Siapkan foto wajah dengan format file .JPG, .JPEG, atau .PNG yang jelas (tidak blur).
            </div>
        """, unsafe_allow_html=True)
    with up_col2:
        st.markdown("""
            <div style="background-color: #FAF8FF; border-left: 4px solid #8A72A4; padding: 15px; border-radius: 8px; min-height: 140px;">
                <strong style="color: #4A3E56;">2. Kondisi Wajah</strong><br><br>Pastikan dahi dan area garis rahang terlihat jelas (tidak tertutup poni tebal atau kain jilbab).
            </div>
        """, unsafe_allow_html=True)
    with up_col3:
        st.markdown("""
            <div style="background-color: #FAF8FF; border-left: 4px solid #8A72A4; padding: 15px; border-radius: 8px; min-height: 140px; margin-bottom: 40px;">
                <strong style="color: #4A3E56;">3. Unggah Foto</strong><br><br>Seret file ke kotak atau klik 'Browse files'. Sistem akan langsung memproses hasil di bawah.
            </div>
        """, unsafe_allow_html=True)

# SECTION 3: VIDEO TUTORIAL 
st.markdown('<div id="video"></div>', unsafe_allow_html=True)
st.markdown('<h3 class="section-title">VIDEO TUTORIAL HIJAB</h3>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">swipe untuk melihat koleksi video tutorial styling hijab lainnya</p>', unsafe_allow_html=True)

list_video = [
    {"link": "https://youtu.be/_rEjIabxTig?si=iWCt7yjw4BHJKSHR"},
    {"link": "https://youtu.be/8f85lC9Y_0U?si=VYHt8J-PEpRhFp0Y"},
    {"link": "https://youtu.be/pWKl8e5JUzY?si=Uw4nk95v6rzIrnYf"},
    {"link": "https://youtu.be/h0BCAHibriI?si=eHxF09nnZC8FHJQY"},
    {"link": "https://youtu.be/R-Fw4Mm0iGg?si=c7NTO1MGt2zSbd2k"}
]

def convert_to_embed_url(url):
    video_id = ""
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    elif "watch?v=" in url:
        video_id = url.split("watch?v=")[1].split("&")[0]
    elif "embed/" in url:
        return url
    
    if video_id:
        return f"https://www.youtube.com/embed/{video_id}"
    return url

video_items_html = ""
for index, item in enumerate(list_video):
    embed_link = convert_to_embed_url(item['link'])
    video_items_html += f"""
    <div style="flex: 0 0 300px; scroll-snap-align: start; background-color: #FFFFFF; border: 1px solid #EFEAEF; border-radius: 12px; padding: 10px; box-shadow: 0px 4px 8px rgba(0,0,0,0.03); text-align: center; box-sizing: border-box;">
        <iframe src="{embed_link}" style="width: 100%; height: 180px; border: none; border-radius: 8px;" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>
    """

full_video_carousel_html = f"""
<div style="display: flex; overflow-x: auto; gap: 15px; padding: 10px 5px; scroll-snap-type: x mandatory; -webkit-overflow-scrolling: touch; background-color: transparent;">
    {video_items_html}
</div>
<style>
    ::-webkit-scrollbar {{ height: 6px; }}
    ::-webkit-scrollbar-track {{ background: #f1f1f1; border-radius: 10px; }}
    ::-webkit-scrollbar-thumb {{ background: #FE81D4; border-radius: 10px; }}
</style>
"""

st.components.v1.html(full_video_carousel_html, height=270, scrolling=False)

st.markdown("<br><hr>", unsafe_allow_html=True)

# SECTION 4: FAQ 
full_faq_html = """
<!-- KOTAK FAQ -->
<div id="faq" class="faq-section-wrapper">
<h3 class="section-title" style="margin-top:0;">FREQUENTLY ASKED QUESTIONS (FAQ)</h3>
<div class="faq-container-inner">
<details class="faq-item">
<summary>Apakah data foto wajah saya disimpan di server?</summary>
<div class="faq-answer">
Tidak. Semua foto diolah secara <i>real-time streaming in-memory</i> menggunakan memori lokal RAM browser dan langsung dihapus saat web ditutup.
</div>
</details>
<details class="faq-item">
<summary>Bagaimana cara mendapatkan hasil analisis bentuk wajah yang paling akurat dengan kamera?</summary>
<div class="faq-answer">
Posisikan wajah tegak lurus (simetris) menatap kamera dengan pencahayaan terang yang merata. Pastikan seluruh batas dahi, pipi, dan rahang terlihat jelas tanpa halangan.
</div>
</details>
<details class="faq-item">
<summary>Apa fungsi dari peta warna bercak merah (Grad-CAM) pada hasil deteksi?</summary>
<div class="faq-answer">
Peta warna Grad-CAM (Explainable AI) berfungsi untuk menunjukkan area visual mana pada wajah (seperti garis rahang atau lebar pipi) yang menjadi fokus utama model AI dalam menyimpulkan bentuk wajah Anda.
</div>
</details>
<details class="faq-item">
<summary>Model hijab yang tertera di rekomendasikan berdasarkan apa? </summary>
<div class="faq-answer">
Model hijab yang tertera adalah model hijab yang sesuai dengan bentuk wajah pengguna dan direkomenadasikan oleh ahli (MUA)
</div>
</details>
</div>
</div> <!-- TUTUP KOTAK FAQ -->

<!-- BUTTON COBA SKRG -->
<div class="cta-banner-box">
<div class="cta-title">Temukan Model Hijab Sesuai Bentuk Wajah Anda</div>
<div class="cta-subtitle">Unggah foto atau gunakan kamera langsung dan temukan model hijab sesuai dengan bentuk wajah Anda!</div>
<a href="#deteksi" class="btn-coba-sekarang">Coba Sekarang</a>
</div>

<!-- FOOTER -->
<div class="faq-footer">
<h5>RekomHijab</h5>
<small>by syifaPrasna © 2026 </small>
</div>
"""

st.markdown(full_faq_html, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)