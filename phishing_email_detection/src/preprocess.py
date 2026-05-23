import re
import html


def clean_email(text: str) -> str:
    """
    Làm sạch nội dung email:
    - Chuyển về chữ thường
    - Xóa HTML tag
    - Chuẩn hóa URL, email, số điện thoại
    - Xóa ký tự đặc biệt
    - Xóa khoảng trắng thừa
    """

    if text is None:
        return ""

    text = str(text)

    # Giải mã HTML entity, ví dụ &amp; -> &
    text = html.unescape(text)

    # Chuyển chữ thường
    text = text.lower()

    # Xóa HTML tag
    text = re.sub(r"<[^>]+>", " ", text)

    # Chuẩn hóa URL
    text = re.sub(r"http\S+|www\S+|https\S+", " URL ", text)

    # Chuẩn hóa địa chỉ email
    text = re.sub(r"\S+@\S+", " EMAIL ", text)

    # Chuẩn hóa số điện thoại đơn giản
    text = re.sub(r"\+?\d[\d\s\-]{7,}\d", " PHONE ", text)

    # Giữ lại chữ, số và khoảng trắng
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_label(label):
    """
    Chuẩn hóa nhãn về 0 hoặc 1.

    1: phishing / spam / malicious
    0: legitimate / ham / safe / normal
    """

    if label is None:
        return None

    label_str = str(label).strip().lower()

    phishing_values = [
        "1",
        "phishing",
        "spam",
        "malicious",
        "fraud",
        "bad",
        "unsafe"
    ]

    safe_values = [
        "0",
        "legitimate",
        "ham",
        "safe",
        "normal",
        "benign",
        "not phishing"
    ]

    if label_str in phishing_values:
        return 1

    if label_str in safe_values:
        return 0

    # Trường hợp label là số nhưng đọc dưới dạng float
    try:
        value = int(float(label_str))
        if value in [0, 1]:
            return value
    except ValueError:
        pass

    return None