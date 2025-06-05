# Hệ thống xếp hạng CV thông minh dựa trên AI

## Giới thiệu
Hệ thống xếp hạng CV thông minh là một ứng dụng tiên tiến sử dụng trí tuệ nhân tạo (AI) để tự động đánh giá và xếp hạng các sơ yếu lý lịch (CV) dựa trên mức độ phù hợp với mô tả công việc (JD). Ứng dụng tận dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP), học máy (Machine Learning - ML) và học sâu (Deep Learning) để tối ưu hóa quy trình tuyển dụng, đặc biệt trong ngành công nghệ thông tin (CNTT). Dự án được phát triển nhằm giải quyết thách thức sàng lọc thủ công tốn thời gian và dễ sai sót.

## Tính năng chính
- **Thu thập dữ liệu:** Thu thập 19,551 cặp CV/JD từ Kaggle, GitHub, Hugging Face, và crawl từ các trang như zety.com, myperfectresume.com.
- **Tiền xử lý:** Loại bỏ email, URL, số điện thoại, chuẩn hóa thuật ngữ với từ điển 300 từ khóa.
- **Trích xuất đặc trưng:** Sử dụng spaCy để trích xuất kỹ năng, kinh nghiệm, học vấn.
- **Vector hóa và gán nhãn:** Vector hóa bằng all-mpnet-base-v2 (768 chiều), tính cosine similarity, tạo tập dữ liệu gán nhãn.
- **Huấn luyện mô hình:** Kết hợp Random Forest (60%), XGBoost (20%), Transformer (20%) trong mô hình Ensemble.
- **Xử lý PDF:** Trích xuất nội dung từ CV PDF bằng pdfplumber, PyMuPDF, Camelot.
- **Xếp hạng CV:** Dự đoán điểm số (0-100), sắp xếp theo thứ tự ưu tiên.
- **Giao diện web:** Tải JD văn bản và CV PDF, hiển thị bảng xếp hạng, biểu đồ điểm số, xuất file Excel/PDF.

## Công nghệ sử dụng
- **Thư viện Python:** Pandas, NumPy, NLTK, spaCy, re, Pickle, Joblib, Sentence Transformers, Scikit-learn, XGBoost, PyTorch, Matplotlib, SciPy, Garbage Collection, Logging, Time, Flask, pdfplumber, PyMuPDF, Camelot, PyPDF2, Selenium, Beautiful Soup.
- **Framework/Web:** Flask, Jinja2, HTML, CSS, JavaScript.
- **Công cụ phát triển:** VSCode, Jupyter Notebook.

## Yêu cầu cài đặt
- **Python:** Phiên bản 3.8 hoặc cao hơn.
- **Thư viện phụ thuộc:** Cài đặt bằng lệnh:
  ```bash
  pip install -r requirements.txt
  ```
- **Dữ liệu:** Tải file `General_Data_CV.csv` và `General_Data_JD.csv` từ thư mục `datasets/` hoặc tự chuẩn bị.

## Hướng dẫn cài đặt và chạy
1. **Clone repository:**
   ```bash
   git clone https://github.com/[username]/NLPResumeRankingAutomatedSystem.git
   cd NLPResumeRankingAutomatedSystem
   ```
2. **Cài đặt môi trường:**
   - Tạo môi trường ảo (tuỳ chọn):
     ```bash
     python -m venv venv
     source venv/bin/activate  # Linux/Mac
     venv\Scripts\activate     # Windows
     ```
   - Cài đặt các thư viện:
     ```bash
     pip install -r requirements.txt
     ```
3. **Chuẩn bị dữ liệu:**
   - Đặt file `General_Data_CV.csv` và `General_Data_JD.csv` vào thư mục `datasets/`.
   - Chạy `setup_data.ipynb` để tạo `labeled_dataset.csv` (nếu cần).
4. **Huấn luyện mô hình:**
   - Chạy `Training_Model_Resume_Ranking.ipynb` để huấn luyện và lưu mô hình (rf_model.pkl, xgb_model.pkl, transformer_model_best.pth).
5. **Chạy ứng dụng web:**
   ```bash
   python app.py
   ```
   - Mở trình duyệt tại `http://localhost:5000` để sử dụng giao diện.

## Cách sử dụng
- Truy cập giao diện web tại `http://localhost:5000`.
- Nhập JD vào ô "Job Description".
- Tải lên CV (PDF, tối đa 200MB) bằng nút "Upload CVs".
- Nhấn "Upload and Rate" để xếp hạng, xem bảng kết quả, biểu đồ điểm số, hoặc xuất file Excel/PDF.
- Nhấn "Reset" để xóa dữ liệu.

## Kết quả nổi bật
- Độ chính xác: MSE = 0.05, R² = 0.92 với mô hình Ensemble.
- Hiệu suất: Xử lý 32 CV trong 6 giây, 1,000 CV trong 4.5 phút.
- Giao diện: Hiển thị bảng xếp hạng, biểu đồ trực quan, hỗ trợ xuất file.

## Đóng góp và phát triển
- **Đóng góp:** Kết hợp mô hình Ensemble, xử lý PDF đa thư viện, giao diện thân thiện.
- **Phát triển:** Tích hợp GPU, hỗ trợ đa ngôn ngữ, triển khai trên cloud (AWS/GCP).

## Giấy phép
Dự án được phát hành dưới giấy phép [MIT License](LICENSE) - xem tệp LICENSE để biết chi tiết.

## Liên hệ
- **Tác giả:** [Tên của bạn]
- **Email:** [Địa chỉ email của bạn]
- **Ngày cập nhật:** 05/06/2025

## Lưu ý
- Đảm bảo dữ liệu đầu vào có chất lượng tốt để đạt hiệu quả tối ưu.
- Báo cáo lỗi hoặc đề xuất cải tiến qua Issues trên GitHub.