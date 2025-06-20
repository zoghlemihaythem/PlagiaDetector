import streamlit as st
import pandas as pd
import time
from plagiarism_detector import run_plagiarism_detection

# For PDF and DOCX text extraction
import fitz  # PyMuPDF
from docx import Document as DocxDocument

st.set_page_config(page_title="Plagiarism Detector", layout="wide")
st.title("📄 Plagiarism Detection Tool")

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True
)

process_mode = st.selectbox(
    "Select Processing Mode",
    options=["Sequential", "Multithreaded", "Multiprocessing", "Producer-Consumer"],
    index=0
)

num_workers = st.slider("Number of Threads/Processes", min_value=1, max_value=16, value=4)
threshold = st.slider("Similarity Threshold (%)", min_value=0, max_value=100, value=70, step=5)


def extract_text_from_pdf(file):
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_text_from_docx(file):
    file.seek(0)
    doc = DocxDocument(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)


if st.button("Run Detection") and uploaded_files:
    documents = []
    filenames = []

    st.info("🔎 Decoding uploaded files...")

    for file in uploaded_files:
        try:
            if file.type == "text/plain":
                # Text file
                try:
                    file.seek(0)
                    content = file.read().decode('utf-8')
                except UnicodeDecodeError:
                    file.seek(0)
                    content = file.read().decode('latin-1')
            elif file.type == "application/pdf":
                content = extract_text_from_pdf(file)
            elif file.type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword"
            ]:
                content = extract_text_from_docx(file)
            else:
                st.warning(f"⚠️ Unsupported file type '{file.type}' for '{file.name}'. Skipping...")
                continue
        except Exception as e:
            st.warning(f"⚠️ Could not read '{file.name}': {str(e)}. Skipping...")
            continue

        documents.append(content)
        filenames.append(file.name)

    if not documents:
        st.error("❌ No valid documents to process after decoding. Please check your files.")
    else:
        start_time = time.time()
        with st.spinner(f"Running plagiarism detection ({process_mode})..."):
            similarity_matrix, report_df = run_plagiarism_detection(
                documents,
                filenames,
                threshold=threshold / 100,
                mode=process_mode.lower().replace("-", "_"),
                num_workers=num_workers
            )
        total_time = time.time() - start_time
        st.success(f"✅ {process_mode} detection completed in {total_time:.4f} seconds")

        st.subheader("⚠️ Potential Plagiarism Pairs")
        suspicious = report_df[report_df['Plagiarism Suspected']]
        st.dataframe(suspicious)

        suspicious_csv = suspicious.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Suspicious Pairs CSV", suspicious_csv, file_name="plagiarism_report_suspicious.csv")

        st.subheader("📥 Download Full Similarity Matrix")
        df_sim = pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)
        sim_csv = df_sim.to_csv(index=True).encode('utf-8')
        st.download_button("📥 Download Full Matrix CSV", sim_csv, file_name="similarity_matrix.csv")

        st.subheader("🔍 Preview of Similarity Matrix (Top 20 Documents Only)")
        preview_size = 20
        preview_df = df_sim.iloc[:preview_size, :preview_size]
        st.dataframe(preview_df.style.format(precision=2))
