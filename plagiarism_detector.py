import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import multiprocessing
import queue
import threading

import fitz  # PyMuPDF for PDFs
from docx import Document as DocxDocument

def extract_text_from_file(file):
    try:
        file.seek(0)
        filetype = file.type

        if filetype == "text/plain":
            try:
                return file.read().decode('utf-8')
            except UnicodeDecodeError:
                file.seek(0)
                return file.read().decode('latin-1')

        elif filetype == "application/pdf":
            file.seek(0)
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text

        elif filetype in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        ]:
            file.seek(0)
            doc = DocxDocument(file)
            paragraphs = [p.text for p in doc.paragraphs]
            return "\n".join(paragraphs)

        else:
            return None
    except Exception:
        return None

def extract_all_texts(files, num_workers=4):
    texts = []
    filenames = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(extract_text_from_file, f): f for f in files}
        for future in concurrent.futures.as_completed(future_to_file):
            f = future_to_file[future]
            text = future.result()
            if text is None or text.strip() == "":
                print(f"Warning: Could not extract text or empty from file '{f.name}'. Skipping.")
            else:
                texts.append(text)
                filenames.append(f.name)

    return texts, filenames

def _compute_similarity(args):
    i, j, similarity_matrix, filenames, threshold = args
    sim = similarity_matrix[i, j]
    return {
        'Document 1': filenames[i],
        'Document 2': filenames[j],
        'Similarity %': round(sim * 100, 2),
        'Plagiarism Suspected': sim >= threshold
    }

def _compute_similarity_batch(args):
    batch_pairs, similarity_matrix, filenames, threshold = args
    results = []
    for i, j in batch_pairs:
        sim = similarity_matrix[i, j]
        results.append({
            'Document 1': filenames[i],
            'Document 2': filenames[j],
            'Similarity %': round(sim * 100, 2),
            'Plagiarism Suspected': sim >= threshold
        })
    return results

def _producer_consumer_worker_with_semaphore(q, similarity_matrix, filenames, threshold, result_list, semaphore):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        i, j = item
        sim = similarity_matrix[i, j]
        result = {
            'Document 1': filenames[i],
            'Document 2': filenames[j],
            'Similarity %': round(sim * 100, 2),
            'Plagiarism Suspected': sim >= threshold
        }
        semaphore.acquire()
        result_list.append(result)
        semaphore.release()
        q.task_done()

def run_plagiarism_detection(documents, filenames, threshold=0.7, mode='sequential', num_workers=4):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    pairs = [(i, j) for i in range(len(documents)) for j in range(i + 1, len(documents))]

    if mode == 'multithreaded':
        chunk_size = max(1, len(pairs) // num_workers)
        batches = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        args_list = [(batch, similarity_matrix, filenames, threshold) for batch in batches]

        report_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_compute_similarity_batch, arg) for arg in args_list]
            for future in concurrent.futures.as_completed(futures):
                report_data.extend(future.result())

    elif mode == 'multiprocessing':
        args_list = [(i, j, similarity_matrix, filenames, threshold) for (i, j) in pairs]
        with multiprocessing.Pool(processes=num_workers) as pool:
            report_data = pool.map(_compute_similarity, args_list)

    elif mode == 'producer_consumer':
        q = queue.Queue()
        for item in pairs:
            q.put(item)
        for _ in range(num_workers):
            q.put(None)

        threads = []
        results = []
        semaphore = threading.Semaphore(1)  # Binary semaphore for critical section

        for _ in range(num_workers):
            t = threading.Thread(target=_producer_consumer_worker_with_semaphore,
                                 args=(q, similarity_matrix, filenames, threshold, results, semaphore))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        report_data = results

    else:  # Sequential
        args_list = [(i, j, similarity_matrix, filenames, threshold) for (i, j) in pairs]
        report_data = [_compute_similarity(arg) for arg in args_list]

    report_df = pd.DataFrame(report_data)
    return similarity_matrix, report_df

# Example usage
if __name__ == "__main__":
    class DummyFile:
        def __init__(self, path, mime_type):
            self.name = path
            self.type = mime_type
        def seek(self, pos): pass
        def read(self):
            with open(self.name, "rb") as f:
                return f.read()

    files = [
        # DummyFile("document1.txt", "text/plain"),
        # DummyFile("document2.pdf", "application/pdf"),
        # DummyFile("document3.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ]

    documents, filenames = extract_all_texts(files, num_workers=4)
    if not documents:
        print("❌ No valid documents extracted.")
    else:
        similarity_matrix, report_df = run_plagiarism_detection(
            documents, filenames, threshold=0.7, mode='producer_consumer', num_workers=4
        )
        print(report_df)
