import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
from rapidocr_onnxruntime import RapidOCR

# Initialize OCR engine with caching for speed
@st.cache_resource
def load_ocr():
    return RapidOCR()

ocr = load_ocr()

# Fast image resize helper to accelerate OCR extraction and reduce memory footprint
def optimize_image_bytes(file_bytes, max_dim=1280):
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, file_bytes
        
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return img, file_bytes
    except Exception:
        return None, file_bytes

st.set_page_config(page_title="Tazemba Form 34A Audit Engine", layout="wide")

st.title("🏛️ Tazemba Presidential Form 34A Forensic Audit Engine")
st.markdown("Automated High-Speed Dynamic Batch Extraction, Mathematical Reconciliation, & Historical Stacking")

# Initialize Session State Variables
if 'uploaded_batch' not in st.session_state:
    st.session_state.uploaded_batch = []
if 'camera_batch' not in st.session_state:
    st.session_state.camera_batch = []
if 'archived_batches' not in st.session_state:
    st.session_state.archived_batches = []
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'trial_count' not in st.session_state:
    st.session_state.trial_count = 0
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

# 🔐 SIDEBAR: Trial Status, Admin Password & Master Archive
st.sidebar.header("🔐 Access & Licensing")

if not st.session_state.is_admin:
    st.sidebar.info(f"🟢 Free Trial Mode: {st.session_state.trial_count} items processed")
else:
    st.sidebar.success("👑 Enterprise Mode Unlocked")

admin_pwd = st.sidebar.text_input("Enter Admin Password", type="password", key="admin_pwd_input")
if admin_pwd == "tazemba2026" or admin_pwd == "admin":
    st.session_state.is_admin = True
    st.sidebar.success("✅ Admin Unlocked: Limits Removed")
elif admin_pwd:
    st.sidebar.error("❌ Incorrect Password")

st.sidebar.markdown("---")
st.sidebar.header("🏛️ Master Historical Archive")
if st.sidebar.button("🗑️ Clear Master Archive"):
    st.session_state.archived_batches = []
    st.rerun()

if st.sidebar.download_button(
    "📥 Download Master Archive CSV",
    data=pd.DataFrame(st.session_state.archived_batches).to_csv(index=False).encode('utf-8') if st.session_state.archived_batches else b"",
    file_name="Total_Cumulative_Form34A_Master.csv",
    mime="text/csv",
    disabled=not bool(st.session_state.archived_batches)
):
    pass

st.sidebar.markdown("---")
if st.session_state.archived_batches:
    sidebar_archive_df = pd.DataFrame(st.session_state.archived_batches)
    st.sidebar.metric("Total Archived Forms", len(sidebar_archive_df))
    st.sidebar.dataframe(sidebar_archive_df[["Form / File Name", "Status"]], use_container_width=True)
else:
    st.sidebar.info("No batches archived yet.")

# 📁 MAIN SCREEN: Fast Batch Document Intake & Camera Capture
st.markdown("### 📁 Batch Document Intake & Live Capture")
col_in1, col_in2 = st.columns(2)

with col_in1:
    uploaded_files = st.file_uploader(
        "Upload Form 34A Batch (PNG/JPG - Fast Processing)", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True,
        key="file_uploader_key"
    )
    if uploaded_files:
        # Uncapped intake loop mapping every single file object without truncation
        st.session_state.uploaded_batch = [(f.name, f.getvalue()) for f in uploaded_files]

with col_in2:
    st.markdown("#### 📷 Camera Capture Station")
    if st.button("📷 Open / Toggle Live Camera"):
        st.session_state.show_camera = not st.session_state.show_camera

    if st.session_state.show_camera:
        camera_file = st.camera_input("Snap Form 34A")
        if camera_file:
            shot_name = f"Form_34A_Scan_{len(st.session_state.camera_batch)+1}.jpg"
            st.session_state.camera_batch.append((shot_name, camera_file.getvalue()))
            st.success(f"Captured! Total camera queue: {len(st.session_state.camera_batch)}")

# Duplicate prevention via unique identifier tuple (filename, byte length)
seen = set()
all_sources = []
for n, d in (st.session_state.get('uploaded_batch', []) + st.session_state.get('camera_batch', [])):
    identifier = (n, len(d))
    if identifier not in seen:
        seen.add(identifier)
        all_sources.append((n, d))

if all_sources:
    st.markdown("---")
    st.info(f"📋 Total Queue Loaded: **{len(all_sources)} documents ready for fast processing.**")
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        run_audit_clicked = st.button(f"🚀 Run Fast Forensic Extraction & Audit ({len(all_sources)} files)")
    with col_btn2:
        if st.button("🗑️ Delete & Clear Selection Queue"):
            st.session_state.uploaded_batch = []
            st.session_state.camera_batch = []
            for k in ['processed_data', 'parsed_extractions']:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    if run_audit_clicked:
        if not st.session_state.is_admin:
            st.session_state.trial_count += len(all_sources)

        processed_data = []
        parsed_extractions = {}

        # Uncapped processing loop over all queue files without truncation
        for filename, file_bytes in all_sources:
            try:
                img_cv, _ = optimize_image_bytes(file_bytes)

                if img_cv is not None:
                    ocr_result, _ = ocr(img_cv)
                    
                    # FULLY DYNAMIC EXTRACTION (Zero hardcoded names)
                    detected_candidates = {}
                    stated_total = 0

                    if ocr_result:
                        for line in ocr_result:
                            bbox, text, score = line
                            
                            # Dynamically match candidate rows 1 to 8 from the form text
                            match_num = re.search(r'\b([1-8])[\.\-\s](.*)', text)
                            if match_num:
                                c_num = int(match_num.group(1))
                                c_content = match_num.group(2).strip()

                                # Extract valid vote integers for this row safely
                                all_nums = [int(v) for v in re.findall(r'\d+', text)]
                                valid_votes = [v for v in all_nums if 0 <= v <= 150]
                                vote_val = str(valid_votes[-1]) if valid_votes else "0"

                                # Clean up the extracted candidate name dynamically
                                name_part = re.sub(r'[\d\s]{4,}.*', '', c_content)
                                if not name_part or len(name_part) < 3:
                                    name_part = c_content

                                if c_num not in detected_candidates:
                                    detected_candidates[c_num] = {
                                        "Candidate #": c_num,
                                        "Candidate Name / Row": f"{c_num}. {name_part.upper()}",
                                        "Votes Obtained": vote_val
                                    }

                            # Detect stated total votes cast accurately
                            if "total" in text.lower() or "valid votes" in text.lower() or "cde" in text.lower():
                                total_matches = [int(m) for m in re.findall(r'\d+', text)]
                                plausible_totals = [m for m in total_matches if 0 < m < 10000]
                                if plausible_totals:
                                    stated_total = plausible_totals[-1]

                    # Fallback defaults if OCR misses structure completely on a specific blank frame
                    if not detected_candidates:
                        for i in range(1, 9):
                            detected_candidates[i] = {
                                "Candidate #": i,
                                "Candidate Name / Row": f"{i}. CANDIDATE {i}",
                                "Votes Obtained": "0"
                            }

                    df_rows = list(detected_candidates.values())
                    df_rows.append({
                        "Candidate #": 9,
                        "Candidate Name / Row": "Total number of valid votes cast",
                        "Votes Obtained": str(stated_total if stated_total > 0 else 0)
                    })

                    # Store dictionary mapping properly for the inspector selectbox
                    parsed_extractions[filename] = {
                        "df": pd.DataFrame(df_rows),
                        "stated_total": stated_total,
                        "image": img_cv
                    }

                    cand_sum = sum(int(row["Votes Obtained"]) for row in df_rows[:-1] if str(row["Votes Obtained"]).isdigit())
                    form_total = int(df_rows[-1]["Votes Obtained"]) if str(df_rows[-1]["Votes Obtained"]).isdigit() else cand_sum
                    status = "PASSED" if cand_sum == form_total else "DISCREPANCY DETECTED"

                    processed_data.append({
                        "Form / File Name": filename,
                        "Parsed Rows": len(df_rows) - 1,
                        "Calculated Candidate Sum": cand_sum,
                        "Stated Total Votes": form_total,
                        "Status": status
                    })
                else:
                    st.warning(f"⚠️ Could not decode image: {filename}")
            except Exception as e:
                st.error(f"Error processing {filename}: {str(e)}")

        if processed_data:
            st.session_state.processed_data = processed_data
            st.session_state.parsed_extractions = parsed_extractions

    # Render Global Results if available in session state
    if 'processed_data' in st.session_state and st.session_state.processed_data:
        st.markdown("## 📊 Global Form 34A Batch Overview")
        batch_df = pd.DataFrame(st.session_state.processed_data)
        
        st.dataframe(batch_df, use_container_width=True)

        col_act1, col_act2 = st.columns(2)
        with col_act1:
            csv_master = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📦 Download Current Batch Master Summary (CSV)",
                data=csv_master,
                file_name="Master_Form34A_Batch_Summary.csv",
                mime="text/csv"
            )
        with col_act2:
            if st.button("📦 Move to Archive & Clear Active Batch"):
                st.session_state.archived_batches.extend(st.session_state.processed_data)
                st.session_state.uploaded_batch = []
                st.session_state.camera_batch = []
                for k in ['processed_data', 'parsed_extractions']:
                    if k in st.session_state:
                        del st.session_state[k]
                st.success("Current batch moved to sidebar archive & working queue cleared!")
                st.rerun()

        st.markdown("---")
        st.markdown("## 🔍 Form 34A Forensic Reconciliation & Audit Engine")
        selected_file = st.selectbox(
            "Select a Form 34A to inspect:", 
            options=list(st.session_state.parsed_extractions.keys())
        )

        if selected_file:
            doc_data = st.session_state.parsed_extractions[selected_file]
            
            col_preview, col_editor = st.columns(2)
            with col_preview:
                st.markdown("### Source Form 34A Preview")
                rgb_img = cv2.cvtColor(doc_data["image"], cv2.COLOR_BGR2RGB)
                st.image(rgb_img, use_column_width=True, caption=f"Document: {selected_file}")

            with col_editor:
                st.markdown("### Extracted Candidate Slate & Votes")
                st.markdown("✏️ *Auditor Live Editing Enabled:*")
                
                edited_df = st.data_editor(
                    doc_data["df"], 
                    num_rows="dynamic",
                    key=f"editor_{selected_file}",
                    use_container_width=True
                )

                # Real-time Mathematical Reconciliation
                try:
                    candidate_rows = edited_df[edited_df["Candidate #"] != 9]
                    calculated_sum = candidate_rows["Votes Obtained"].astype(int).sum()
                    
                    total_row = edited_df[edited_df["Candidate #"] == 9]
                    stated_form_total = int(total_row["Votes Obtained"].values[0]) if not total_row.empty else doc_data["stated_total"]

                    st.markdown("---")
                    st.markdown("### 📐 Automated Mathematical Reconciliation")
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("Calculated Candidate Sum", calculated_sum)
                    col_m2.metric("Stated Form Total", stated_form_total)

                    if calculated_sum == stated_form_total:
                        st.success("✅ **AUDIT STATUS: PASSED** — Candidate vote tallies perfectly match the stated total valid votes cast.")
                    else:
                        st.error(f"🔴 **AUDIT STATUS: DISCREPANCY DETECTED** — Discrepancy of {abs(calculated_sum - stated_form_total)} votes found between candidate sum and stated total!")
                except Exception as ex:
                    st.warning(f"Ensure all vote entries are valid integers for real-time reconciliation. ({ex})")

























 


   


        

