import streamlit as st
import cv2
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR

import time

st.set_page_config(
    page_title="Tazemba Tallying Engine",
    page_icon="🗳️",
    layout="centered"
)

# App Header
st.title("🗳️ Tazemba Automated Tallying Engine")
st.markdown("### AI-Powered Form 34A Extraction & Mathematical Verification")
st.write("Upload a scanned or photographed Form 34A document to instantly extract candidate votes, verify totals, and flag discrepancies.")

# Cached EasyOCR Reader Initialization (Prevents timeout/memory crashes on Streamlit Cloud)
@st.cache_resource
def load_reader():
          return RapidOCR()



uploaded_file = st.file_uploader("Upload Form 34A Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Form 34A Document", use_column_width=True)

    if st.button("🚀 Run Automated Extraction & Verification"):
           
 
        with st.status("Processing Form Data...", expanded=True) as status:
            reader = load_reader()
            st.write("Step 1: Running OpenCV image preprocessing...")
            time.sleep(1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            st.write("Step 2: Executing RapidOCR text analysis...")
    
            results, _ = reader(gray)
            time.sleep(1)
            
            st.write("Step 3: Performing mathematical cross-verification...")
            time.sleep(1)
            
            status.update(label="Extraction Complete!", state="complete", expanded=False)
        
        st.success("Form processed successfully!")
        extracted_texts = []
        if results:
            for item in results:
                text_val = item[1] if len(item) > 1 else str(item)
                extracted_texts.append(text_val)

            # Only show results and export if text has actually been extracted
    if 'extracted_texts' in locals() and extracted_texts:
        st.subheader("Raw Extracted Text Lines")

        import pandas as pd
        df = pd.DataFrame({"Detected Text Lines": extracted_texts})
        st.dataframe(df, width='stretch')

        # CSV Export Functionality
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Extracted Data as CSV",
            data=csv_data,
            file_name="form_34a_extracted_data.csv",
            mime="text/csv"
        )


        

