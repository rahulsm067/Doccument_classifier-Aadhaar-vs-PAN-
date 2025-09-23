import streamlit as st
from PIL import Image
import os
from time import sleep
from src.cnn_classifier import CNNClassifier

# Initialize classifier
classifier = CNNClassifier(model_path="models/cnn_model.pth")

# Streamlit UI
st.set_page_config(page_title="Document Classifier", layout="wide")
st.title("📄 Document Classifier: Aadhaar vs PAN ")

uploaded_files = st.file_uploader(
    "Upload Document Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} document(s).")

    if st.button("Classify Document(s)"):
        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                # Save temporarily
                save_path = f"temp_{idx}.jpg"
                img = Image.open(uploaded_file)
                
                
                if img.mode == 'RGBA':
                    # Create a white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    # Paste the RGBA image onto the white background using alpha channel as mask
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode not in ['RGB', 'L']:
                    # Convert any other mode to RGB
                    img = img.convert('RGB')
                
                img.save(save_path, 'JPEG')

                st.image(img, caption=f"Document {idx}", width='stretch')

                # Show progress
                progress_text = f"Classifying Document {idx}/{len(uploaded_files)}..."
                my_bar = st.progress(0, text=progress_text)

                # CNN prediction
                label, conf, infer_time = classifier.predict(save_path)
                sleep(0.5)
                my_bar.progress(100, text=f"Finished Document {idx}")

                # Show results
                if conf < 0.6:
                    st.warning(f"⚠️ Document {idx}: Low confidence prediction.")
                else:
                    st.success(f"✅ Document {idx}: {label}")
                    st.info(f"Confidence: {conf:.2f}")
                    st.info(f"Inference Time: {infer_time:.2f} sec")

                # Clean up
                os.remove(save_path)

            except Exception as e:
                st.error(f"Error processing Document {idx}: {e}")
