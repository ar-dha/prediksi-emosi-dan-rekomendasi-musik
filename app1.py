import streamlit as st
import numpy as np
import pandas as pd
import cv2
from keras.models import load_model

# Load model dan data musik
classifier = load_model('deteksi_emosi.h5', compile=False)
mood_music = pd.read_csv("data_moods.csv")

def recommend_songs(pred_class):
    if pred_class in ['disgust', 'sadness']:
        mood = 'Sad'
    elif pred_class == 'happiness':
        mood = 'Happy'
    elif pred_class in ['fear', 'anger']:
        mood = 'Calm'
    elif pred_class in ['surprise', 'neutral']:
        mood = 'Energetic'
    else:
        return None

    recommended_songs = mood_music[mood_music['mood'] == mood]
    if recommended_songs.empty:
        recommended_songs = mood_music
    recommended_songs = recommended_songs.sample(frac=1).reset_index(drop=True)
    recommended_songs = recommended_songs.head(5)
    return recommended_songs

# Fungsi untuk mengubah nilai prediksi menjadi label
def map_prediction_to_emotion(prediction):
    emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    max_prob_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[max_prob_index]
    return predicted_emotion

# Fungsi untuk capture gambar dari kamera
def capture_frame():
    # Initialize the camera
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        st.error("Tidak dapat mengakses kamera. Silakan coba lagi.")
        return None

    # Capture a frame
    ret, frame = capture.read()
    if ret:
        # Preprocess the frame for prediction
        frame_resized = cv2.resize(frame, (224, 224))
        frame_resized_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        st.image(frame_resized_bgr, caption='Gambar Terambil.', use_column_width=True)
        return frame_resized
    else:
        st.error("Gagal mengambil gambar. Silakan coba lagi.")
        return None
    # Release the camera
    capture.release()

# Judul pada Streamlit
st.title('Prediksi Emosi dan Rekomendasi Musik')

# Pilihan untuk mengunggah atau mengambil gambar
option = st.radio("Pilih opsi:", ("Pilih gambar", "Ambil gambar dari kamera"))

if option == "Pilih gambar":
    # Unggah gambar
    uploaded_file = st.file_uploader('Pilih sebuah gambar...', type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        # Membaca gambar
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        st.image(image_bgr, caption='Gambar Terunggah.', use_column_width=True)
        st.write("")
        st.write("Memproses...")

        # Prediksi emosi
        image_resized = cv2.resize(image, (224, 224))
        image_expanded = np.expand_dims(image_resized, axis=0)
        image_array = np.array(image_expanded)
        predictions = classifier.predict(image_array)
        predicted_emotion = map_prediction_to_emotion(predictions)

        # Memunculkan prediksi emosi
        st.write(f'Prediksi Emosi: {predicted_emotion}')
        
        # Rekomendasi musik berdasarkan prediksi emosi
        recommended_songs = recommend_songs(predicted_emotion)
        if recommended_songs is not None:
            st.write('Rekomendasi Musik:')
            for index, row in recommended_songs.iterrows():
                st.write(f'{index + 1}. **{row["name"]}** by {row["artist"]}')
                spotify_link = f'[Buka di Spotify](spotify:track:{row["id"]})'
                st.markdown(spotify_link, unsafe_allow_html=True)
        else:
            st.write('Tidak ada rekomendasi musik untuk emosi ini.')
            
        st.markdown("<p style='text-align: center;'>Silakan klik 'R' untuk mendapatkan rekomendasi musik lainnya.</p>", unsafe_allow_html=True)


elif option == "Ambil gambar dari kamera":
    st.write("Klik tombol untuk mengambil gambar")
    if st.button("Ambil Gambar"):
        captured_frame = capture_frame()
        if captured_frame is not None:
            height, width, _ = captured_frame.shape

            # ROI (region of interest)
            zoom_factor = 1.3  
            zoomed_width = int(width / zoom_factor)
            zoomed_height = int(height / zoom_factor)
            roi = captured_frame[
                int((height - zoomed_height) / 2):int((height + zoomed_height) / 2),
                int((width - zoomed_width) / 2):int((width + zoomed_width) / 2)
            ]
            # Prediksi emosi dari gambar yang diambil
            image_resized = cv2.resize(roi, (224, 224))
            image_expanded = np.expand_dims(image_resized, axis=0)
            image_array = np.array(image_expanded)
            predictions = classifier.predict(image_array)
            predicted_emotion = map_prediction_to_emotion(predictions)

            # Memunculkan prediksi emosi
            st.write(f'Prediksi Emosi: {predicted_emotion}')
            
            # Rekomendasi musik berdasarkan prediksi emosi
            recommended_songs = recommend_songs(predicted_emotion)
            if recommended_songs is not None:
                st.write('Rekomendasi Musik:')
                for index, row in recommended_songs.iterrows():
                    st.write(f'{index + 1}. **{row["name"]}** by {row["artist"]}')
                    spotify_link = f'[Buka di Spotify](spotify:track:{row["id"]})'
                    st.markdown(spotify_link, unsafe_allow_html=True)
            else:
                st.write('Tidak ada rekomendasi musik untuk emosi ini.')
            
            st.markdown("<p style='text-align: center;'>Silakan klik 'R' untuk mendapatkan rekomendasi musik lainnya.</p>",unsafe_allow_html=True) 
