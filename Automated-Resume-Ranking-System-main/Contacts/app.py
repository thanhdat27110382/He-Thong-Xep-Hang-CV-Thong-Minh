import os
import joblib
import re
import pandas as pd
from flask import Flask, request, render_template, jsonify
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import logging

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Tải stopwords
nltk.download('stopwords')
stop_words_en = set(stopwords.words('english'))
try:
    stop_words_vi = set(stopwords.words('vietnamese.txt'))
except:
    stop_words_vi = set()
    logging.warning("Không tìm thấy stopwords tiếng Việt, sử dụng tập rỗng.")

app = Flask(__name__)

# Định nghĩa đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'resume_ranking_model.pkl')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'csvfiles', 'ketquaxephang', 'resume_ranking_results.xlsx')

# Tải mô hình
try:
    model = joblib.load(MODEL_PATH)
    logging.info(f"Đã tải mô hình từ {MODEL_PATH}")
except FileNotFoundError:
    logging.error(f"Tệp mô hình {MODEL_PATH} không tồn tại.")
    model = None

# Đảm bảo thư mục uploads tồn tại
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    logging.info(f"Đã tạo thư mục uploads: {UPLOAD_DIR}")

def clean_text(text):
    """Làm sạch văn bản."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    text = ' '.join(word for word in words if word not in stop_words_en and word not in stop_words_vi)
    logging.debug(f"Văn bản sau khi làm sạch: {text[:100]}...")
    return text

def extract_text_from_resume(resume_path):
    """Trích xuất văn bản từ PDF resume."""
    text = ""
    try:
        with open(resume_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        logging.info(f"Đã trích xuất văn bản từ {resume_path}, độ dài: {len(text)} ký tự")
    except Exception as e:
        logging.error(f"Lỗi khi trích xuất văn bản từ {resume_path}: {e}")
    cleaned_text = clean_text(text.strip())
    if not cleaned_text:
        logging.warning(f"Không trích xuất được văn bản từ {resume_path}")
    return cleaned_text

def calculate_cosine_similarity(resume_text, job_description):
    """Tính độ tương đồng cosine."""
    try:
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        texts = [job_description, resume_text]
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        score = similarity * 100
        logging.debug(f"Điểm cosine similarity: {score}")
        return round(score, 2), 100
    except Exception as e:
        logging.error(f"Lỗi khi tính cosine similarity: {e}")
        return 0.0, 100

def get_suggestion(score):
    """Trả về gợi ý dựa trên điểm số."""
    try:
        if score > 80:
            suggestion = "🔥 Excellent match! Your resume is well-optimized."
        elif score > 60:
            suggestion = "✅ Good match! Consider adding more relevant keywords."
        else:
            suggestion = "⚡ Low match! Try improving your skills section and adding industry-specific terms."
        logging.debug(f"Gợi ý AI cho điểm {score}: {suggestion}")
        return suggestion
    except Exception as e:
        logging.error(f"Lỗi khi tạo gợi ý: {e}")
        return "N/A"

@app.route('/')
def index():
    return render_template('indexRanking.html', results=None, top_match=None)

@app.route('/upload', methods=['POST'])
def upload():
    logging.info("Nhận yêu cầu upload")
    if 'resumes' not in request.files or 'job_description' not in request.form:
        logging.error("Thiếu resumes hoặc job_description")
        return jsonify({"error": "Vui lòng cung cấp ít nhất một resume và mô tả công việc."}), 400

    resume_files = request.files.getlist('resumes')
    job_description = clean_text(request.form['job_description'])

    if not resume_files or not job_description:
        logging.error("Resume hoặc mô tả công việc không hợp lệ")
        return jsonify({"error": "Resume hoặc mô tả công việc không hợp lệ."}), 400

    # Kiểm tra tổng kích thước file
    total_size = sum(f.content_length for f in resume_files if f) / (1024 * 1024)
    if total_size > 200:
        logging.error(f"Tổng kích thước file {total_size}MB vượt quá 200MB")
        return jsonify({"error": "Tổng kích thước file vượt quá 200MB."}), 400

    results = []
    for resume_file in resume_files:
        if resume_file.filename == '' or not resume_file.filename.endswith('.pdf'):
            logging.warning(f"Bỏ qua file không hợp lệ: {resume_file.filename}")
            continue

        # Lưu resume
        resume_path = os.path.join(UPLOAD_DIR, resume_file.filename)
        resume_file.save(resume_path)
        logging.info(f"Đã lưu resume: {resume_path}")

        # Trích xuất và làm sạch văn bản
        resume_text = extract_text_from_resume(resume_path)
        if not resume_text:
            logging.warning(f"Không trích xuất được văn bản từ {resume_file.filename}")
            continue

        # Tính điểm cosine similarity
        score, max_score = calculate_cosine_similarity(resume_text, job_description)

        # Tạo gợi ý AI
        suggestion = get_suggestion(score)

        # Dự đoán danh mục
        category = "N/A"
        if model:
            try:
                category = model.predict([resume_text])[0]
                logging.info(f"Dự đoán danh mục cho {resume_file.filename}: {category}")
            except Exception as e:
                logging.error(f"Lỗi khi dự đoán danh mục cho {resume_file.filename}: {e}")
        else:
            logging.warning("Mô hình không được tải, sử dụng danh mục mặc định: N/A")

        # Lưu kết quả
        result = {
            'resume_file': resume_file.filename,
            'score': score,
            'suggestion': suggestion,
            'category': category,
            'job_description': job_description[:100]
        }
        results.append(result)
        logging.debug(f"Kết quả cho {resume_file.filename}: {result}")

    # Sắp xếp theo điểm số giảm dần
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    logging.info(f"Tổng số resume được xử lý: {len(results)}")

    # Lưu vào Excel
    try:
        result_df = pd.DataFrame(results)
        if os.path.exists(OUTPUT_PATH):
            try:
                existing_df = pd.read_excel(OUTPUT_PATH)
                result_df = pd.concat([existing_df, result_df], ignore_index=True)
            except Exception as e:
                logging.error(f"Lỗi khi đọc file Excel hiện có: {e}")
        result_df.to_excel(OUTPUT_PATH, index=False)
        logging.info(f"Đã lưu kết quả vào {OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu Excel: {e}")

    # Xác định top match
    top_match = results[0]['resume_file'] if results else None
    logging.info(f"Top match: {top_match}")

    return render_template('indexRanking.html', results=results, top_match=top_match)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)