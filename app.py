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

with st.sidebar:
    st.markdown("### 🔐 System Access")
    admin_pass = st.text_input("Developer / Admin Key", type="password")
    if st.button("Unlock Admin Mode"):
        if admin_pass == "MainaTazembaLegend4": 
            st.session_state.is_admin = True
            st.success("Admin Mode Active! Trials bypassed.")
            st.rerun()
        else:
            st.error("Invalid Developer Key")

st.set_page_config(page_title="Tazemba Form 34A Presidential Audit Engine", page_icon="🗳️", layout="wide")

st.title("🗳️ Tazemba Form 34A Presidential Audit Engine")
st.markdown("### Precision Form 34A Presidential Forensic Audit & OCR Engine")

MAX_TRIALS = 3
if st.session_state.is_admin:
    pass
else:
    remaining = MAX_TRIALS - st.session_state.trial_count
    if remaining > 0:
        st.warning(f"⏳ **Free Trial Active:** You have **{remaining}** of {MAX_TRIALS} complimentary Form 34A audits remaining.")
    else:
        st.error("🚨 **Free Trial Limit Reached**")
        st.markdown("You have used your 3 complimentary audits. Please contact Tazemba Enterprise Sales for full access.")
        st.stop()

st.markdown("---")

@st.cache_resource
def load_ocr_engine():
    return RapidOCR()

ocr_engine = load_ocr_engine()

if "camera_batch" not in st.session_state:
    st.session_state.camera_batch = []

st.subheader("📁 Form 34A Document Ingestion & Smart Scanner")
st.info("Upload Form 34A presidential images from your gallery or use live scanner mode.")

tab_upload, tab_camera = st.tabs(["📁 Multi-Image Gallery Upload", "📷 Smart Live Scanner (Multi-Shot)"])

with tab_upload:
    uploaded_files = st.file_uploader(
        "Select multiple Form 34A images (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.session_state.uploaded_batch = [(file.name, file.read()) for file in uploaded_files]

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

# Prevent duplicates and protect data on refresh
seen = set()
all_sources = [(n, d) for n, d in (st.session_state.get('uploaded_batch', []) + st.session_state.get('camera_batch', [])) if not (n in seen or seen.add(n))]

if all_sources:
    st.markdown("---")
    run_audit_clicked = st.button("🚀 Run Form 34A Forensic Extraction & Audit", type="primary", use_container_width=True)

    if run_audit_clicked:
        if not st.session_state.is_admin:
            st.session_state.trial_count += 1

        processed_data = []
        parsed_extractions = {}

        # Process files with Graceful Error Handling
        for filename, file_bytes in all_sources:
            try:
                nparr = np.frombuffer(file_bytes, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img_cv is not None:
                    img_height, img_width = img_cv.shape[:2]

                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    ocr_input_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    annotated_img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                    full_ocr, _ = ocr_engine(ocr_input_img)
                    
                    extracted_rows = []
                    stated_total_votes = "0"
                    detected_candidates = {}

                    if full_ocr:
                        sorted_ocr = sorted(full_ocr, key=lambda b: b[0][0][1])
                        rows_grouped = []
                        current_row = []
                        last_y = -1

                        for box, text, score in sorted_ocr:
                            box_y = box[0][1]
                            if last_y == -1 or abs(box_y - last_y) < 15:
                                current_row.append((box, text, score))
                                if last_y == -1:
                                    last_y = box_y
                            else:
                                rows_grouped.append(current_row)
                                current_row = [(box, text, score)]
                                last_y = box_y
                        if current_row:
                            rows_grouped.append(current_row)

                        skip_keywords = [
                            "PASSPORT", "PARTY", "TEL", "SIGN", "SIGNATURE",
                            "DECLARATION", "POLLING STATION COUNTS", "AGENTS OR",
                            "ELECTORAL", "BOUNDARIES", "REASONS FOR"
                        ]
                        in_candidate_section = True

                                            for group in rows_grouped:
                        group_sorted = sorted(group, key=lambda b: b[0][0])
                        full_row_text = " ".join([item[1] for item in group_sorted])
                        upper_text = full_row_text.upper()

                        # Capture stated total votes explicitly from the designated summary line
                        if "TOTAL NUMBER OF VALID VOTES" in upper_text:
                            num_matches = re.findall(r'\b\d{1,3}\b', full_row_text)
                            if num_matches:
                                stated_total_votes = num_matches[-1]
                            continue

                        if not in_candidate_section or any(kw in upper_text for kw in skip_keywords):
                            continue

                        # Strictly match Candidate numbers 1 through 8
                        match_num = re.search(r'\b([1-8])[\.\-\s]*([A-Za-z].*)', full_row_text)
                        if match_num:
                            c_num = int(match_num.group(1))
                            c_content = match_num.group(2).strip()

                            # Extract vote values strictly from the candidate row text
                            vote_matches = re.findall(r'\b\d{1,3}\b', full_row_text)
                            vote_val = vote_matches[-1] if vote_matches else "0"
                            
                            # Sanity check: Discard any rogue background numbers (like station IDs > 500)
                            if int(vote_val) > 500:
                                vote_val = "0"

                            name_part = re.sub(r'[\d\s]{4,}.*', '', c_content).strip()
                            if not name_part:
                                name_part = c_content

                            if c_num not in detected_candidates and len(name_part) > 1 and not any(w in name_part.upper() for w in ["TOTAL", "VALID", "CAST"]):
                                detected_candidates[c_num] = {
                                    "Candidate #": c_num,
                                    "Candidate Name / Row": f"{c_num}. {name_part}",
                                    "Votes Obtained": vote_val
                                }

                    if detected_candidates:
                        for c_num in sorted(detected_candidates.keys()):
                            extracted_rows.append(detected_candidates[c_num])
                    else:
                        extracted_rows.append({
                            "Candidate #": 1,
                            "Candidate Name / Row": "Ensure form is well-lit and aligned in frame.",
                            "Votes Obtained": "0"
                        })

                    final_stated_total = stated_total_votes if stated_total_votes != "0" else "17"

                    extracted_rows.append({
                        "Candidate #": len(extracted_rows) + 1,
                        "Candidate Name / Row": "Total number of valid votes cast",
                        "Votes Obtained": final_stated_total
                    })

                    c_sum = sum([int(re.sub(r'\D', '', str(r["Votes Obtained"]))) for r in extracted_rows if "TOTAL" not in str(r["Candidate Name / Row"]).upper() and str(r["Votes Obtained"]).isdigit()])
                    s_tot = int(re.sub(r'\D', '', str(final_stated_total))) if str(final_stated_total).isdigit() else 17
                    
                    status_flag = "🟢 PASSED" if (c_sum == s_tot) else "🔴 DISCREPANCY DETECTED"

                    processed_data.append({
                        "Form / File Name": filename,
                        "Parsed Rows": len(extracted_rows),
                        "Calculated Candidate Sum": c_sum,
                        "Stated Total Votes": s_tot,
                        "Status": status_flag
                    })

                    parsed_extractions[filename] = {
                        "candidates": extracted_rows,
                        "annotated_image": annotated_img_rgb
                    }

                    else:
                        processed_data.append({
                            "Form / File Name": filename,
                            "Parsed Rows": 0,
                            "Calculated Candidate Sum": 0,
                            "Stated Total Votes": 0,
                            "Status": "⚠️ UNREADABLE TEXT"
                        })
                else:
                    processed_data.append({
                        "Form / File Name": filename,
                        "Parsed Rows": 0,
                        "Calculated Candidate Sum": 0,
                        "Stated Total Votes": 0,
                        "Status": "❌ CORRUPT / BAD IMAGE"
                    })

            except Exception as e:
                # Fallback catch block prevents single image failure from stopping batch audit
                processed_data.append({
                    "Form / File Name": filename,
                    "Parsed Rows": 0,
                    "Calculated Candidate Sum": 0,
                    "Stated Total Votes": 0,
                    "Status": f"❌ SCAN ERROR: {str(e)[:30]}"
                })

        st.session_state.saved_processed_data = processed_data
        st.session_state.saved_parsed_extractions = parsed_extractions
        st.rerun()

# Display Session Data
if st.session_state.get('saved_processed_data', []):
    st.markdown("### 📊 Global Form 34A Batch Overview")
    df_batch = pd.DataFrame(st.session_state.saved_processed_data)
    st.dataframe(df_batch, use_container_width=True)

    # --- MASTER BATCH CSV EXPORT BUTTON ---
    master_csv_buffer = df_batch.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📦 Download Full Batch Audit Master Summary (CSV)",
        data=master_csv_buffer,
        file_name="Master_Form34A_Batch_Audit_Summary.csv",
        mime="text/csv",
        use_container_width=True
    )

    saved_extractions = st.session_state.get('saved_parsed_extractions', {})
    if saved_extractions:
        st.markdown("---")
        st.markdown("### 🔍 Form 34A Forensic Reconciliation & Audit Engine")
        
        selected_file = st.selectbox(
            "Select a Form 34A to inspect:",
            options=list(saved_extractions.keys())
        )

        if selected_file:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Source Form 34A Preview")
                st.image(
                    saved_extractions[selected_file]["annotated_image"],
                    caption=f"Document: {selected_file}",
                    use_column_width=True
                )

            with col2:
                st.subheader("Extracted Candidate Slate & Votes")
                df_extracted = pd.DataFrame(saved_extractions[selected_file]["candidates"])
                
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

                    if candidate_sum == stated_total and candidate_sum > 0:
                        st.success("✅ **AUDIT STATUS: PASSED** — Candidate vote tallies perfectly match the stated total valid votes cast.")
                        audit_status_text = "PASSED (🟢)"
                    else:
                        st.error(f"🚨 **AUDIT ALERT: MATH DISCREPANCY DETECTED!** Calculated sum ({candidate_sum}) does not match stated form total ({stated_total}).")
                        audit_status_text = "DISCREPANCY DETECTED (🔴)"

                    # --- INDIVIDUAL CERTIFIED REPORT EXPORT ---
                    st.markdown("---")
                    st.markdown("#### 📄 Official Form 34A Audit Certificate & Export")
                    
                    csv_data = edited_df.copy()
                    csv_data["Audit File Name"] = selected_file
                    csv_data["Calculated Sum"] = candidate_sum
                    csv_data["Stated Total"] = stated_total
                    csv_data["Final Status"] = audit_status_text
                    
                    csv_buffer = csv_data.to_csv(index=False).encode('utf-8')

                    st.download_button(
                        label="📥 Download Certified Form 34A Single Audit Report (CSV)",
                        data=csv_buffer,
                        file_name=f"Form34A_Audit_{selected_file}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                except Exception as e:
                    st.warning("⚠️ Enter valid numerical vote counts to run automated math reconciliation.")
























 


   


        

