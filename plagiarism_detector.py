import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import multiprocessing
import queue
import threading

def _compute_similarity(args):
    i, j, similarity_matrix, filenames, threshold = args
    sim = similarity_matrix[i, j]
    return {
        'Document 1': filenames[i],
        'Document 2': filenames[j],
        'Similarity %': round(sim * 100, 2),
        'Plagiarism Suspected': sim >= threshold
    }

def _producer_consumer_worker(q, similarity_matrix, filenames, threshold):
    report_data = []
    while True:
        item = q.get()
        if item is None:
            break
        i, j = item
        sim = similarity_matrix[i, j]
        report_data.append({
            'Document 1': filenames[i],
            'Document 2': filenames[j],
            'Similarity %': round(sim * 100, 2),
            'Plagiarism Suspected': sim >= threshold
        })
        q.task_done()
    q.task_done()
    return report_data

def run_plagiarism_detection(documents, filenames, threshold=0.7, mode='sequential', num_workers=4):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    pairs = [(i, j) for i in range(len(documents)) for j in range(i + 1, len(documents))]
    args_list = [(i, j, similarity_matrix, filenames, threshold) for (i, j) in pairs]

    if mode == 'multithreaded':
        report_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_compute_similarity, arg) for arg in args_list]
            for future in concurrent.futures.as_completed(futures):
                report_data.append(future.result())

    elif mode == 'multiprocessing':
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
        for _ in range(num_workers):
            t = threading.Thread(target=lambda res=results: res.extend(_producer_consumer_worker(q, similarity_matrix, filenames, threshold)))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        report_data = results

    else:  # Sequential (default)
        report_data = [_compute_similarity(arg) for arg in args_list]

    report_df = pd.DataFrame(report_data)
    return similarity_matrix, report_df
