import argparse
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.preprocess import clean_email, normalize_label


def load_dataset(path, text_col, label_col):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file dataset: {path}")

    df = pd.read_csv(path)

    print("Các cột trong dataset:")
    print(df.columns.tolist())

    if text_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột nội dung email: {text_col}")

    if label_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột nhãn: {label_col}")

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    # Xóa dòng rỗng
    df = df.dropna(subset=["text", "label"])

    # Chuẩn hóa label
    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Làm sạch email
    df["clean_text"] = df["text"].apply(clean_email)

    # Xóa email quá ngắn
    df = df[df["clean_text"].str.len() > 5]

    return df


def build_models():
    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words="english"
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ]),

        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words="english"
            )),
            ("clf", MultinomialNB())
        ]),

        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words="english"
            )),
            ("clf", LinearSVC(class_weight="balanced"))
        ])
    }

    return models


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print(f"Kết quả mô hình: {name}")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Safe", "Phishing"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "model": model
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train mô hình phát hiện email phishing"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/emails.csv",
        help="Đường dẫn dataset CSV"
    )

    parser.add_argument(
        "--text_col",
        type=str,
        default="text",
        help="Tên cột chứa nội dung email"
    )

    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Tên cột chứa nhãn"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/phishing_model.pkl",
        help="Đường dẫn lưu model"
    )

    args = parser.parse_args()

    print("Đang đọc dữ liệu...")
    df = load_dataset(args.data, args.text_col, args.label_col)

    print("\nThông tin dataset sau xử lý:")
    print(df["label"].value_counts())
    print(f"Tổng số mẫu: {len(df)}")

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = build_models()

    results = []

    for name, model in models.items():
        print(f"\nĐang huấn luyện mô hình: {name}")
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)

    # Chọn mô hình có F1-score cao nhất
    best_result = max(results, key=lambda x: x["f1"])
    best_model = best_result["model"]

    print("\n" + "#" * 60)
    print(f"Mô hình tốt nhất: {best_result['name']}")
    print(f"F1-score: {best_result['f1']:.4f}")
    print("#" * 60)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    model_package = {
        "model": best_model,
        "model_name": best_result["name"],
        "accuracy": best_result["accuracy"],
        "precision": best_result["precision"],
        "recall": best_result["recall"],
        "f1": best_result["f1"]
    }

    joblib.dump(model_package, args.output)

    print(f"\nĐã lưu mô hình tại: {args.output}")


if __name__ == "__main__":
    main()