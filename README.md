# Phishing Email Detection

Demo phát hiện email phishing bằng Machine Learning, sử dụng Python, scikit-learn và Streamlit.

## Chức năng

- Huấn luyện mô hình phân loại email an toàn / phishing.
- Tiền xử lý nội dung email.
- Vector hóa văn bản bằng TF-IDF.
- So sánh các mô hình:
  - Logistic Regression
  - Naive Bayes
  - Linear SVM
- Lưu mô hình tốt nhất theo F1-score.
- Demo giao diện web bằng Streamlit.
- Hiển thị điểm rủi ro và các dấu hiệu đáng nghi trong email.

## Cấu trúc thư mục

```text
phishing_email_detection/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── emails.csv
├── models/
│   └── phishing_model.pkl
└── src/
    ├── preprocess.py
    └── predict_utils.py
