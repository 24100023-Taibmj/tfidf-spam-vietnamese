import streamlit as st
import joblib

st.set_page_config(
    page_title="Phân loại tin nhắn Spam",
    page_icon="📩",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("spam_model.joblib")

model = load_model()

st.title("📩 Phân loại tin nhắn Spam tiếng Việt")
st.write(
    "Ứng dụng sử dụng TF-IDF kết hợp Naive Bayes "
    "để phân loại tin nhắn."
)

message = st.text_area(
    "Nhập nội dung tin nhắn:",
    placeholder="Ví dụ: Chúc mừng bạn trúng thưởng...",
    height=150
)

if st.button("Phân loại tin nhắn", type="primary"):
    if not message.strip():
        st.warning("Vui lòng nhập nội dung tin nhắn.")
    else:
        prediction = int(model.predict([message])[0])
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        spam_probability = float(
            probabilities[classes.index(1)]
        )

        st.subheader("Kết quả phân loại")

        if prediction == 1:
            st.error("⚠️ Tin nhắn được dự đoán là SPAM")
        else:
            st.success(
                "✅ Tin nhắn được dự đoán là BÌNH THƯỜNG"
            )

        st.write(
            f"Điểm dự đoán Spam của mô hình: "
            f"{spam_probability:.2%}"
        )

        st.caption(
            "Kết quả chỉ mang tính tham khảo; "
            "mô hình có thể phân loại sai."
        )

st.divider()
st.caption(
    "Đề tài: Tìm hiểu TF-IDF và ứng dụng "
    "trong phân loại tin nhắn Spam tiếng Việt."
)
