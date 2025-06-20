import streamlit as st
import pandas as pd
import numpy as np
import time
from plagiarism_detector import run_plagiarism_detection

st.set_page_config(page_title="Plagiarism Detector", layout="wide")
st.title("📄 Plagiarism Detection Tool")

uploaded_files = st.file_uploader("Upload your .txt documents", type="txt", accept_multiple_files=True)

process_mode = st.selectbox(
    "Select Processing Mode",
    options=["Sequential", "Multithreaded", "Multiprocessing", "Producer-Consumer"],
    index=0
)

num_workers = st.slider("Number of Threads/Processes", min_value=1, max_value=16, value=4)

threshold = st.slider("Similarity Threshold (%)", min_value=0, max_value=100, value=70, step=5)

if st.button("Run Detection") and uploaded_files:
    documents = []
    filenames = []

    st.info("🔎 Decoding uploaded files...")

    for file in uploaded_files:
        try:
            content = file.read().decode('utf-8')
        except UnicodeDecodeError:
            try:
                content = file.read().decode('latin-1')
            except UnicodeDecodeError:
                st.warning(f"⚠️ File '{file.name}' could not be decoded. Skipping...")
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

        # Provide full similarity matrix as a download instead of displaying it
        st.subheader("📥 Download Full Similarity Matrix")
        df_sim = pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)
        sim_csv = df_sim.to_csv(index=True).encode('utf-8')
        st.download_button("📥 Download Full Matrix CSV", sim_csv, file_name="similarity_matrix.csv")

        # Display only a small preview of the similarity matrix for large datasets
        st.subheader("🔍 Preview of Similarity Matrix (Top 20 Documents Only)")
        preview_size = 20
        preview_df = df_sim.iloc[:preview_size, :preview_size]
        st.dataframe(preview_df.style.format(precision=2))
