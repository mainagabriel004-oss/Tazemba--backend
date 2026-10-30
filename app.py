import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import re
from rapidocr_onnxruntime import RapidOCR

# ==========================================
# TRIAL & ADMIN BYPASS SETUP
# ==========================================
if "trial_count" not in st.session_state:
    st.session_state.trial_count = 0

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# Admin Sidebar Login
with st.sidebar:
    st.markdown("### 🔐 System Access")
    admin_pass = st.text_input("Developer / Admin Key", type="password")
    if admin_pass == "MainaTazembaLegend4": 
        st.session_state.is_admin = True
        st.success("Admin Mode Active")
# ==========================================

# Page Config & Branding
st.set_page_config(
    page_title="Tazemba Form 34A Presidential Audit Engine",
    page_icon="🗳️",
    layout="wide"
)

st.title("🗳️ Tazemba Form 34A Presidential Audit Engine")
st.markdown("### Precision Form 34A Presidential Forensic Audit & OCR Engine")

# ==========================================
# MAIN PAGE TRIAL COUNTER & GATE
# ==========================================
MAX_TRIALS = 3
if st.session_state.is_admin:
    st.info("🔓 **Admin Mode Active:** Unlimited trials enabled.")
else:
    remaining = MAX_TRIALS - st.session_state.trial_count
    if remaining > 0:
        st.warning(f"⏳ **Free Trial Active:** You have **{remaining}** of {MAX_TRIALS} complimentary Form 34A audits remaining.")
    else:
        st.error("🚨 **Free Trial Limit Reached**")
        st.markdown("You have used your 3 complimentary audits. Please contact Tazemba Enterprise Sales for full access.")
        st.stop()
# ==========================================

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
st.subheader("📁 Form 34A Document Ingestion & Smart Scanner")
st.info("Upload Form 34A presidential images from your gallery or use live scanner mode.")

tab_upload, tab_camera = st.tabs(["📁 Multi-Image Gallery Upload", "📷 Smart Live Scanner (Multi-Shot)"])

image_files = []

# Mode 1: Multi-Image Gallery Upload
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
        bytes_data = camera_photo.read()
        nparr = np.frombuffer(bytes_data, np.uint8)
        test_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if test_img is not None:
            gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            col_g1, col_g2 = st.columns(2)
            col_g1.metric("Lighting Level", f"{int(brightness)} / 255")
            col_g2.metric("Edge Sharpness", f"{int(sharpness)}")

            if brightness > 70 and sharpness > 100:
                st.success("🟢 **Quality Check Passed:** Optimal lighting and clarity.")
            else:
                st.warning("⚠️ **Warning:** Adjust lighting or hold device steady for better OCR extraction.")

            if st.button("➕ Add Captured Photo to Batch Queue", use_container_width=True):
                shot_name = f"Form_34A_Scan_{len(st.session_state.camera_batch) + 1}.jpg"
                st.session_state.camera_batch.append((shot_name, bytes_data))
                st.success(f"Added! Total queued: {len(st.session_state.camera_batch)}")

    if st.session_state.camera_batch:
        st.markdown("---")
        st.markdown(f"**Current Camera Batch Queue:** {len(st.session_state.camera_batch)} form(s).")
        if st.button("🗑️ Clear Camera Queue", type="secondary"):
            st.session_state.camera_batch = []
            st.rerun()

all_sources = image_files + st.session_state.camera_batch

# --- CORE 100% DYNAMIC EXTRACTION PIPELINE ---
processed_data = []
parsed_extractions = {}

if all_sources:
    st.markdown("---")
    st.success(f"Successfully loaded {len(all_sources)} Form 34A(s). Running 100% dynamic optical extraction...")

    for filename, file_bytes in all_sources:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv is not None:
            img_height, img_width = img_cv.shape[:2]

            # --- ADVANCED MOBILE IMAGE PREPROCESSING ---
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            processed_img = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            ocr_input_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
            # ---------------------------------------------

            annotated_img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            # Run OCR across the full document dynamically
            full_ocr, _ = ocr_engine(ocr_input_img)

            extracted_rows = []
            stated_total_votes = "0"

            if full_ocr:
                sorted_ocr = sorted(full_ocr, key=lambda r: r[0][0][1])
                
                rows_grouped = []
                current_row = []
                last_y = -1

                for item in sorted_ocr:
                    box, text, score = item
                    box_y = box[0][1]

                    if last_y == -1 or abs(box_y - last_y) < 25:
                        current_row.append(item)
                        if last_y == -1:
                            last_y = box_y
                    else:
                        rows_grouped.append(current_row)
                        current_row = [item]
                        last_y = box_y
                if current_row:
                    rows_grouped.append(current_row)

                row_counter = 1
                for group in rows_grouped:
                    group_sorted = sorted(group, key=lambda r: r[0][0][0])
                    full_row_text = " ".join([item[1] for item in group_sorted]).strip()
                    upper_text = full_row_text.upper()

                    # Skip general metadata headers and non-candidate labels
                    if any(w in upper_text for w in ["INDEPENDENT ELECTORAL", "PRESIDENTIAL ELECTION RESULTS", "STATION NAME", "REGISTRATION", "POLLING STATION", "WARD", "CONSTITUENCY", "COUNTY", "CODE", "NAME OF CANDIDATE", "NO. OF VALID"]):
                        continue

                    # Capture stated total votes dynamically
                    if "TOTAL" in upper_text or "VALID VOTES" in upper_text:
                        num_matches = re.findall(r'\d+', full_row_text)
                        if num_matches:
                            stated_total_votes = num_matches[-1]
                        continue

                    # Dynamically capture row content and associated vote tally
                    if len(full_row_text) > 3:
                        num_matches = re.findall(r'\d+', full_row_text)
                        vote_val = num_matches[-1] if num_matches else "0"
                        
                        name_part = re.sub(r'\s+' + vote_val + r'$', '', full_row_text).strip()
                        if not name_part:
                            name_part = full_row_text

                        extracted_rows.append({
                            "Candidate #": row_counter,
                            "Candidate Name / Row": name_part,
                            "Votes Obtained": vote_val if vote_val != full_row_text else "0"
                        })
                        row_counter += 1

            if not extracted_rows:
                extracted_rows.append({
                    "Candidate #": 1,
                    "Candidate Name / Row": "No text lines detected. Please check form lighting or alignment.",
                    "Votes Obtained": "0"
                })
            else:
                # Append Official Total Valid Votes Row at the bottom
                extracted_rows.append({
                    "Candidate #": len(extracted_rows) + 1,
                    "Candidate Name / Row": "Total number of valid votes cast",
                    "Votes Obtained": stated_total_votes
                })

            processed_data.append({
                "Form / File Name": filename,
                "Parsed Rows": len(extracted_rows),
                "Status": "🟢 DYNAMIC SCAN READY"
            })

            parsed_extractions[filename] = {
                "candidates": extracted_rows,
                "annotated_image": annotated_img_rgb
            }

    # Global Batch Dashboard View
    if processed_data:
        st.markdown("### 📊 Global Form 34A Batch Overview")
        st.dataframe(pd.DataFrame(processed_data), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔍 Form 34A Forensic Reconciliation & Audit Engine")
        
        selected_file = st.selectbox(
            "Select a Form 34A to inspect:",
            options=list(parsed_extractions.keys())
        )

        if selected_file:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Source Form 34A Preview")
                st.image(
                    parsed_extractions[selected_file]["annotated_image"],
                    caption=f"Document: {selected_file}",
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

                        if "TOTAL" in row_text or "VALID VOTES" in row_text:
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
                        st.error(f"🚨 **AUDIT ALERT: MATH DISCREPANCY DETECTED!** Calculated sum ({candidate_sum}) does not match stated form total ({stated_total}).")
                        audit_status_text = "DISCREPANCY DETECTED (🔴)"

                    # --- CERTIFIED REPORT EXPORT ---
                    st.markdown("---")
                    st.markdown("#### 📄 Official Form 34A Audit Certificate & Export")
                    
                    csv_data = edited_df.copy()
                    csv_data["Audit File Name"] = selected_file
                    csv_data["Calculated Sum"] = candidate_sum
                    csv_data["Stated Total"] = stated_total
                    csv_data["Final Status"] = audit_status_text
                    
                    csv_buffer = csv_data.to_csv(index=False).encode('utf-8')

                    st.download_button(
                        label="📥 Download Certified Form 34A Audit Report (CSV)",
                        data=csv_buffer,
                        file_name=f"Form34A_Audit_{selected_file}.csv",
                        mime="text/css",
                        use_container_width=True
                    )

                except Exception as e:
                    st.warning("⚠️ Enter valid numerical vote counts to run automated math reconciliation.")



















 


   


        

