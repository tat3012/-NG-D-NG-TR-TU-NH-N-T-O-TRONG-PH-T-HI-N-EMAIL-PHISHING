import os
import joblib
import streamlit as st

from src.preprocess import clean_email
from src.predict_utils import (
    extract_email_features,
    get_risk_level,
    get_prediction_score
)


MODEL_PATH = "models/phishing_model.pkl"


st.set_page_config(
    page_title="Phishing Email Detection",
    page_icon="📧",
    layout="wide"
)


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    model_package = joblib.load(MODEL_PATH)
    return model_package


def show_header():
    st.title("📧 Phát hiện Email Phishing bằng Học máy")
    st.write(
        """
        Ứng dụng demo sử dụng kỹ thuật **Machine Learning** để phân loại email thành:
        **Email an toàn** hoặc **Email phishing**.
        """
    )


def show_sidebar(model_package):
    st.sidebar.title("Thông tin mô hình")

    if model_package is None:
        st.sidebar.error("Chưa tìm thấy model.")
        st.sidebar.write("Hãy train model trước bằng lệnh:")
        st.sidebar.code("python train_model.py")
    else:
        st.sidebar.success("Đã tải model thành công")

        st.sidebar.write("**Model:**", model_package.get("model_name", "Unknown"))
        st.sidebar.write("**Accuracy:**", round(model_package.get("accuracy", 0), 4))
        st.sidebar.write("**Precision:**", round(model_package.get("precision", 0), 4))
        st.sidebar.write("**Recall:**", round(model_package.get("recall", 0), 4))
        st.sidebar.write("**F1-score:**", round(model_package.get("f1", 0), 4))

    st.sidebar.markdown("---")
    st.sidebar.write("### Hướng dẫn")
    st.sidebar.write(
        """
        1. Dán nội dung email vào ô nhập.
        2. Nhấn nút **Phân tích email**.
        3. Xem kết quả dự đoán và mức độ rủi ro.
        """
    )


def example_emails():
    safe_email = """Hello John,

Please find attached the meeting agenda for tomorrow.
We will discuss the project timeline and next week's tasks.

Best regards,
Project Team
"""

    phishing_email = """URGENT: Your bank account has been suspended!

We detected unusual login activity on your account.
Please click the link below to verify your password immediately:
http://fake-bank-security-login.com/verify

If you do not verify within 24 hours, your account will be locked.
"""

    return safe_email, phishing_email


def main():
    model_package = load_model()

    show_header()
    show_sidebar(model_package)

    if model_package is None:
        st.warning("Bạn cần huấn luyện mô hình trước khi chạy demo.")
        st.code("python train_model.py")
        return

    model = model_package["model"]

    safe_email, phishing_email = example_emails()

    st.subheader("Nhập nội dung email cần kiểm tra")

    sample_option = st.selectbox(
        "Chọn email mẫu hoặc tự nhập:",
        [
            "Tự nhập",
            "Email an toàn mẫu",
            "Email phishing mẫu"
        ]
    )

    if sample_option == "Email an toàn mẫu":
        default_text = safe_email
    elif sample_option == "Email phishing mẫu":
        default_text = phishing_email
    else:
        default_text = ""

    email_text = st.text_area(
        "Nội dung email:",
        value=default_text,
        height=250,
        placeholder="Dán nội dung email vào đây..."
    )

    analyze_button = st.button("🔍 Phân tích email")

    if analyze_button:
        if not email_text.strip():
            st.warning("Vui lòng nhập nội dung email.")
            return

        cleaned_text = clean_email(email_text)

        prediction, phishing_score = get_prediction_score(model, cleaned_text)
        risk_level, risk_message = get_risk_level(phishing_score)
        features = extract_email_features(email_text)

        st.markdown("---")
        st.subheader("Kết quả phân tích")

        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == 1:
                st.error("Kết luận: PHISHING")
            else:
                st.success("Kết luận: AN TOÀN")

        with col2:
            st.metric(
                label="Điểm rủi ro phishing",
                value=f"{phishing_score * 100:.2f}%"
            )

        with col3:
            if risk_level == "Cao":
                st.error(f"Mức rủi ro: {risk_level}")
            elif risk_level == "Trung bình":
                st.warning(f"Mức rủi ro: {risk_level}")
            else:
                st.success(f"Mức rủi ro: {risk_level}")

        st.info(risk_message)

        st.markdown("---")
        st.subheader("Dấu hiệu được phát hiện trong email")

        feature_col1, feature_col2, feature_col3 = st.columns(3)

        with feature_col1:
            st.metric("Số lượng URL", features["url_count"])

        with feature_col2:
            st.metric("Số email address", features["email_count"])

        with feature_col3:
            st.metric("Độ dài email", features["length"])

        feature_col4, feature_col5 = st.columns(2)

        with feature_col4:
            st.metric("Số dấu !", features["exclamation_count"])

        with feature_col5:
            st.metric("Số dấu ?", features["question_count"])

        suspicious_words = features["found_suspicious_words"]

        if suspicious_words:
            st.warning(
                "Từ khóa đáng nghi phát hiện được: "
                + ", ".join(suspicious_words)
            )
        else:
            st.success("Không phát hiện từ khóa đáng nghi phổ biến.")

        with st.expander("Xem nội dung email sau tiền xử lý"):
            st.write(cleaned_text)

        st.markdown("---")
        st.subheader("Giải thích ngắn gọn")

        if prediction == 1:
            st.write(
                """
                Mô hình dự đoán email này có khả năng là phishing vì nội dung có thể chứa
                các dấu hiệu như yêu cầu xác minh tài khoản, đường link lạ, ngôn ngữ khẩn cấp
                hoặc các từ khóa liên quan đến đăng nhập, mật khẩu, tài khoản ngân hàng.
                """
            )
        else:
            st.write(
                """
                Mô hình dự đoán email này có vẻ an toàn hơn. Tuy nhiên, kết quả chỉ mang tính
                hỗ trợ. Người dùng vẫn cần kiểm tra người gửi, đường link, file đính kèm và
                ngữ cảnh thực tế trước khi tin tưởng email.
                """
            )


if __name__ == "__main__":
    main()