import re


def extract_email_features(raw_text):
    """
    Trích xuất một số dấu hiệu thủ công để hiển thị trên web app.
    Các feature này dùng để giải thích thêm, không nhất thiết dùng cho model.
    """

    text = str(raw_text).lower()

    url_count = len(re.findall(r"http\S+|www\S+|https\S+", text))
    email_count = len(re.findall(r"\S+@\S+", text))
    exclamation_count = text.count("!")
    question_count = text.count("?")

    suspicious_words = [
        "urgent",
        "verify",
        "password",
        "login",
        "account",
        "bank",
        "limited",
        "suspended",
        "click",
        "confirm",
        "security",
        "update",
        "winner",
        "prize",
        "free",
        "payment",
        "invoice",
        "locked"
    ]

    found_words = []
    for word in suspicious_words:
        if word in text:
            found_words.append(word)

    return {
        "url_count": url_count,
        "email_count": email_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "found_suspicious_words": found_words,
        "length": len(text)
    }


def get_risk_level(score):
    """
    Chuyển điểm rủi ro thành mức Low / Medium / High.
    score nằm trong khoảng 0 -> 1.
    """

    if score >= 0.75:
        return "Cao", "Email có nhiều khả năng là phishing."
    elif score >= 0.45:
        return "Trung bình", "Email có dấu hiệu đáng nghi, cần kiểm tra kỹ."
    else:
        return "Thấp", "Email có vẻ an toàn hơn, nhưng vẫn cần thận trọng."


def get_prediction_score(model, text):
    """
    Lấy xác suất phishing nếu model hỗ trợ predict_proba.
    Nếu không có predict_proba, dùng decision_function.
    """

    prediction = model.predict([text])[0]

    # Logistic Regression và Naive Bayes có predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]

        # Xác suất thuộc lớp 1 - phishing
        phishing_score = float(proba[1])

    # LinearSVC không có predict_proba
    elif hasattr(model, "decision_function"):
        decision_score = model.decision_function([text])[0]

        # Chuyển decision score về khoảng 0-1 tương đối
        phishing_score = 1 / (1 + pow(2.71828, -decision_score))

    else:
        phishing_score = 1.0 if prediction == 1 else 0.0

    return int(prediction), phishing_score