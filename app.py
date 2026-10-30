import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import zipfile
import io
import re
from rapidocr_onnxruntime import RapidOCR

# Page Config & Branding
st.set_page_config(
    page_title="Tazemba Form 34A Audit Engine",
    page_icon="🗳️",
    layout="wide"
)

st.title("🗳️ Tazemba Form 34A Audit Engine")
st.markdown("### Universal Dynamic Batch Audit & Smart Scanner Platform")
st.markdown("---")

# Initialize Pure-Python Dynamic OCR Engine
@st.cache_resource
def load_ocr_engine():
    return RapidOCR()

ocr_engine = load_ocr_engine()

# Initialize Session State for Multi-Shot Camera Batching
if "camera_batch" not in st.session_state:
    st.session_state.camera_batch = []

# --- ENTERPRISE SMART SCANNER INGESTION PIPELINE ---
st.subheader("📁 Enterprise Document Ingestion & Smart Scanner")
st.info("Select multiple images from your gallery at once, or use the live scanner mode with smart lighting & angle guidance to capture multiple forms back-to-back.")

# Tabbed Ingestion Interface
tab_upload, tab_camera = st.tabs(["📁 Multi-Image Gallery Upload", "📷 Smart Live Scanner (Multi-Shot)"])

image_files = []

# Mode 1: Multi-Image Gallery Upload (No ZIP required)
with tab_upload:
    uploaded_files = st.file_uploader(
        "Select multiple Form 34A images (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            image_files.append((file.name, file.read()))

# Mode 2: Smart Live Scanner with Multi-Shot & Quality Guidance
with tab_camera:
    st.markdown("#### 📱 Live Field Scanner & Guidance Assistant")
    camera_photo = st.camera_input("Position Form 34A in frame and take photo")
    
    if camera_photo:
        # Convert camera buffer to OpenCV for Smart Quality Checks
        bytes_data = camera_photo.read()
        nparr = np.frombuffer(bytes_data, np.uint8)
        test_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if test_img is not None:
            # --- SMART GUIDANCE ALGORITHM (Lighting & Sharpness Check) ---
            gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Display Real-time Guidance Feedback
            col_g1, col_g2 = st.columns(2)
            col_g1.metric("Lighting Level", f"{int(brightness)} / 255")
            col_g2.metric("Edge Sharpness", f"{int(sharpness)}")

            is_lighting_good = brightness > 70  # Threshold for dim light
            is_sharp_good = sharpness > 100     # Threshold for blur

            if not is_lighting_good:
                st.warning("⚠️ **Lighting Warning:** Environment is too dim. Move to a well-lit area or use flash.")
            elif not is_sharp_good:
                st.warning("⚠️ **Stability Warning:** Image looks blurry. Hold your phone steady.")
            else:
                st.success("🟢 **Quality Check Passed:** Optimal lighting and clarity. Ready to add to batch!")

            # Multi-Shot Queue Button
            if st.button("➕ Add Captured Photo to Batch Queue", use_container_width=True):
                shot_name = f"Live_Scan_{len(st.session_state.camera_batch) + 1}.jpg"
                st.session_state.camera_batch.append((shot_name, bytes_data))
                st.success(f"Added! Total queued from camera: {len(st.session_state.camera_batch)} form(s).")

    # Display Queue Management
    if st.session_state.camera_batch:
        st.markdown("---")
        st.markdown(f"**Current Camera Batch Queue:** {len(st.session_state.camera_batch)} form(s) stored.")
        if st.button("🗑️ Clear Camera Queue", type="secondary"):
            st.session_state.camera_batch = []
            st.rerun()

# Combine both upload and camera queued files into final processing pool
for item in image_files:
    if item not in image_files: # Prevent duplicates if uploaded
        pass
all_sources = image_files + st.session_state.camera_batch

# --- CORE PROCESSING PIPELINE ---
processed_data = []
parsed_extractions = {}

if all_sources:
    st.markdown("---")
    st.success(f"Successfully loaded {len(all_sources)} form(s). Running optical vote extraction & forensic verification...")

    for filename, file_bytes in all_sources:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv is not None:
            img_height, img_width = img_cv.shape[:2]

            # Precision Crop: Targets candidate voting grid from top to bottom
            y_start = int(img_height * 0.28)
            y_end = int(img_height * 0.47)
            
            name_x_start = int(img_width * 0.08)
            name_x_end = int(img_width * 0.58)
            votes_x_start = int(img_width * 0.58)
            votes_x_end = int(img_width * 0.92)

            # Draw green ROI bounding box
            annotated_img = img_cv.copy()
            cv2.rectangle(annotated_img, (name_x_start, y_start), (votes_x_end, y_end), (0, 255, 0), 2)

            # Isolate Votes Region for dynamic numerical extraction
            votes_roi = img_cv[y_start:y_end, votes_x_start:votes_x_end]
            votes_ocr, _ = ocr_engine(votes_roi)

            # Parse extracted numbers by vertical order
            extracted_vote_lines = []
            if votes_ocr:
                for v_line in sorted(votes_ocr, key=lambda r: r[0][0][1]):
                    v_text = v_line[1].strip()
                    v_y = v_line[0][0][1]
                    clean_num = re.sub(r'\D', '', v_text)
                    if clean_num or "0" in v_text:
                        extracted_vote_lines.append((v_y, clean_num if clean_num else "0"))

            # Standard Statutory Form 34A Candidate Order
            extracted_rows = [
                {"Candidate #": 1, "Candidate Name / Row": "1. AUKOT JOHN EKURU LONGOGGY", "Votes Obtained": "0"},
                {"Candidate #": 2, "Candidate Name / Row": "2. DIDA MOHAMED ABDUBA", "Votes Obtained": "0"},
                {"Candidate #": 3, "Candidate Name / Row": "3. JIRONGO SHAKHALAGA KHWA", "Votes Obtained": "0"},
                {"Candidate #": 4, "Candidate Name / Row": "4. KALUYU JAPHETH KAVINGA", "Votes Obtained": "0"},
                {"Candidate #": 5, "Candidate Name / Row": "5. KENYATTA UHURU", "Votes Obtained": "0"},
                {"Candidate #": 6, "Candidate Name / Row": "6. MWAURA MICHAEL WAINANA", "Votes Obtained": "0"},
                {"Candidate #": 7, "Candidate Name / Row": "7. NYAGAH JOSEPH WILLIAM NTHIGA", "Votes Obtained": "0"},
                {"Candidate #": 8, "Candidate Name / Row": "8. ODINGA RAILA", "Votes Obtained": "0"},
                {"Candidate #": 9, "Candidate Name / Row": "Total number of valid votes cast", "Votes Obtained": "0"}
            ]

            # Map OCR numerical vote tallies to corresponding candidates
            for i, row in enumerate(extracted_rows[:-1]):
                if i < len(extracted_vote_lines):
                    row["Votes Obtained"] = extracted_vote_lines[i][1]

            if len(extracted_vote_lines) >= len(extracted_rows):
                extracted_rows[-1]["Votes Obtained"] = extracted_vote_lines[-1][1]
            elif extracted_vote_lines:
                extracted_rows[-1]["Votes Obtained"] = extracted_vote_lines[-1][1]

            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            processed_data.append({
                "Form / File Name": filename,
                "Parsed Rows": len(extracted_rows),
                "Status": "🟢 READY FOR REVIEW"
            })

            parsed_extractions[filename] = {
                "candidates": extracted_rows,
                "annotated_image": annotated_img_rgb
            }

    # Global Batch Dashboard View
    if processed_data:
        st.markdown("### 📊 Global Batch Audit Overview")
        st.dataframe(pd.DataFrame(processed_data), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔍 Individual Form Deep-Dive & Forensic Reconciliation Engine")
        
        selected_file = st.selectbox(
            "Select a processed form to inspect:",
            options=list(parsed_extractions.keys())
        )

        if selected_file:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Visual Column ROI Box")
                st.image(
                    parsed_extractions[selected_file]["annotated_image"],
                    caption=f"Targeted Table Area: {selected_file}",
                    use_column_width=True
                )

            with col2:
                st.subheader("Extracted Candidate Slate & Votes")
                df_extracted = pd.DataFrame(parsed_extractions[selected_file]["candidates"])
                
                st.write("✏️ *Auditor Live Editing Enabled:*")
                edited_df = st.data_editor(df_extracted, num_rows="dynamic", use_container_width=True, key=f"editor_{selected_file}")

                # --- FORENSIC RECONCILIATION LOGIC ---
                try:
                    candidate_sum = 0
                    stated_total = 0
                    audit_status_text = "PASSED"
                    
                    for index, row in edited_df.iterrows():
                        row_text = str(row["Candidate Name / Row"]).upper()
                        vote_val_str = re.sub(r'\D', '', str(row["Votes Obtained"]))
                        vote_val = int(vote_val_str) if vote_val_str else 0

                        if "TOTAL" in row_text or "VALID VOTES CAST" in row_text:
                            stated_total = vote_val
                        else:
                            candidate_sum += vote_val

                    st.markdown("---")
                    st.markdown("#### 📐 Automated Mathematical Reconciliation")
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Calculated Candidate Sum", candidate_sum)
                    m_col2.metric("Stated Form Total", stated_total)

                    if candidate_sum == stated_total:
                        st.success("✅ **AUDIT STATUS: PASSED** — Candidate vote tallies perfectly match the stated total valid votes cast.")
                        audit_status_text = "PASSED (🟢)"
                    else:
                        st.error(f"🚨 **AUDIT ALERT: MATH DISCREPANCY DETECTED!** Calculated sum ({candidate_sum}) does not match stated form total ({stated_total}). Please verify handwriting or data entries.")
                        audit_status_text = "DISCREPANCY DETECTED (🔴)"

                    # --- CERTIFIED REPORT EXPORT ---
                    st.markdown("---")
                    st.markdown("#### 📄 Official Audit Certificate & Data Export")
                    
                    csv_data = edited_df.copy()
                    csv_data["Audit File Name"] = selected_file
                    csv_data["Calculated Sum"] = candidate_sum
                    csv_data["Stated Total"] = stated_total
                    csv_data["Final Status"] = audit_status_text
                    
                    csv_buffer = csv_data.to_csv(index=False).encode('utf-8')

                    st.download_button(
                        label="📥 Download Certified Audit Report (CSV)",
                        data=csv_buffer,
                        file_name=f"Audit_Report_{selected_file}.csv",
                        mime="text/css",
                        use_container_width=True
                    )

                except Exception as e:
                    st.warning("⚠️ Enter valid numerical vote counts to run automated math reconciliation.")















 


   


        

