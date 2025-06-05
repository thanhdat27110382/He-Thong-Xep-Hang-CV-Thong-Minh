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

# Tải stopwords
nltk.download('stopwords')
stop_words_en = set(stopwords.words('english'))
# Giả lập stopwords tiếng Việt nếu không có file vietnamese.txt
stop_words_vi = set(stopwords.words('vietnamese.txt'))

app = Flask(__name__)

# Định nghĩa đường dẫn tương đối
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'resume_ranking_model.pkl')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'csvfiles', 'ketquaxephang', 'resume_ranking_results.xlsx')

# Tải mô hình đã huấn luyện
try:
    model = joblib.load(MODEL_PATH)
    print(f"Đã tải mô hình từ {MODEL_PATH}")
except FileNotFoundError:
    print(f"Lỗi: Tệp {MODEL_PATH} không tồn tại.")
    model = None

# Đảm bảo thư mục uploads tồn tại
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def clean_text(text):
    """Làm sạch văn bản."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    text = ' '.join(word for word in words if word not in stop_words_en and word not in stop_words_vi)
    return text

def extract_text_from_resume(resume_path):
    """Trích xuất văn bản từ PDF resume."""
    text = ""
    try:
        with open(resume_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Lỗi khi trích xuất văn bản từ {resume_path}: {e}")
    return clean_text(text.strip())

def calculate_cosine_similarity(resume_text, job_description):
    """Tính độ tương đồng cosine."""
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    texts = [job_description, resume_text]
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    score = similarity * 100  # Chuẩn hóa thành thang 100
    return round(score, 2)

def get_ai_suggestion(score):
    """Trả về gợi ý AI dựa trên điểm số."""
    if score > 80:
        return "🔥 Excellent match! Your resume is well-optimized."
    elif score > 60:
        return "✅ Good match! Consider adding more relevant keywords."
    else:
        return "⚡ Low match! Try improving your skills section and adding industry-specific terms."

@app.route('/')
def index():
    return render_template('indexRanking.html', results=None, top_match=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'resumes' not in request.files or 'job_description' not in request.form:
        return jsonify({"error": "Vui lòng cung cấp ít nhất một resume và mô tả công việc."}), 400

    resume_files = request.files.getlist('resumes')
    job_description = clean_text(request.form['job_description'])

    if not resume_files or not job_description:
        return jsonify({"error": "Resume hoặc mô tả công việc không hợp lệ."}), 400

    # Kiểm tra tổng kích thước và định dạng
    total_size = 0
    max_size = 200 * 1024 * 1024  # 200MB
    results = []

    for resume_file in resume_files:
        if resume_file.filename == '':
            continue
        if not resume_file.filename.endswith('.pdf'):
            return jsonify({"error": f"File {resume_file.filename} không phải PDF."}), 400
        total_size += len(resume_file.read())
        resume_file.seek(0)  # Reset con trỏ file
        if total_size > max_size:
            return jsonify({"error": "Tổng kích thước file vượt quá 200MB."}), 400

        # Lưu resume
        resume_path = os.path.join(UPLOAD_DIR, resume_file.filename)
        resume_file.save(resume_path)

        # Trích xuất và làm sạch văn bản
        resume_text = extract_text_from_resume(resume_path)
        if not resume_text:
            return jsonify({"error": f"Không thể trích xuất văn bản từ {resume_file.filename}."}), 400

        # Tính điểm cosine similarity
        score = calculate_cosine_similarity(resume_text, job_description)

        # Lưu kết quả
        results.append({
            'resume_file': resume_file.filename,
            'score': score,
            'ai_suggestion': get_ai_suggestion(score)
        })

    # Sắp xếp theo điểm số giảm dần
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Lưu vào Excel
    result_df = pd.DataFrame(results)
    if os.path.exists(OUTPUT_PATH):
        try:
            existing_df = pd.read_excel(OUTPUT_PATH)
            result_df = pd.concat([existing_df, result_df], ignore_index=True)
        except Exception as e:
            print(f"Lỗi khi đọc file Excel hiện có: {e}")
    result_df.to_excel(OUTPUT_PATH, index=False)

    # Xác định top match
    top_match = results[0]['resume_file'] if results else None

    return render_template('indexRanking.html', results=results, top_match=top_match)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)