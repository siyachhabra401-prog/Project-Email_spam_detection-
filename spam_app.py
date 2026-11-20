import pandas as pd
import numpy as np
import pickle
import streamlit as st

st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

# ---------- Styling ----------
st.markdown("""
<style>
.header {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    padding: 28px 20px;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 20px;
}
.header h1 { color: white; margin: 0; font-size: 2.2rem; }
.header p  { color: #e0e7ff; margin: 6px 0 0 0; font-size: 1.05rem; }
.result-spam {
    background: #fee2e2; color: #b91c1c;
    padding: 22px; border-radius: 14px; text-align: center;
    font-size: 1.6rem; font-weight: 700; border: 2px solid #ef4444;
}
.result-ok {
    background: #dcfce7; color: #15803d;
    padding: 22px; border-radius: 14px; text-align: center;
    font-size: 1.6rem; font-weight: 700; border: 2px solid #22c55e;
}
div.stButton > button {
    width: 100%; background: #4f46e5; color: white;
    font-size: 1.1rem; font-weight: 600; padding: 0.6rem 0; border-radius: 10px; border: none;
}
div.stButton > button:hover { background: #4338ca; color: white; }
</style>
""", unsafe_allow_html=True)


def main():
    model = pickle.load(open("spam_model.pkl","rb"))

    # ---------- Header ----------
    st.markdown("""
    <div class="header">
        <h1>📧 Email Spam Detector</h1>
        <p>Enter the email details and find out if it is spam or not</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Sidebar ----------
    st.sidebar.title("About")
    st.sidebar.info("This app uses a Machine Learning model trained on 7,000 emails to predict whether an email is **Spam** or **Not Spam**.")
    st.sidebar.markdown("**Spam emails usually have:**")
    st.sidebar.markdown("- many links\n- many special characters\n- many capital words\n- longer length")

    # ---------- Encoding (same numbers as the notebook) ----------
    subjects = {"Security Alert":0,"Win Prize":1,"Invoice":2,"Meeting":3,
                "Offer":4,"Project Update":5,"Greetings":6,"Account Verification":7}
    domains = {"outlook.com":0,"yahoo.com":1,"gmail.com":2,"company.com":3,"unknown":4}

    # ---------- Inputs ----------
    st.subheader("✉️ Email Details")

    col1, col2 = st.columns(2)

    with col1:
        s1 = st.selectbox("📌 Subject", list(subjects.keys()))
        s3 = st.selectbox("🌐 Sender domain", list(domains.keys()))
        s2 = st.radio("📎 Has attachment?", ("No","Yes"), horizontal=True)

    with col2:
        p2 = st.number_input("📏 Email length", 0.0, 1000.0, 100.0, step=1.0)
        p4 = st.number_input("❗ Special characters", 0, 100, 3, step=1)
        p5 = st.number_input("🔠 Capital words", 0, 100, 3, step=1)

    p3 = st.slider("🔗 Number of links in the email", 0, 15, 0)

    p1 = subjects[s1]
    p7 = domains[s3]
    p6 = 1 if s2 == "Yes" else 0

    data_new = pd.DataFrame({
    'Subject':p1,
    'Email_Length':p2,
    'Num_Links':p3,
    'Num_Special_Chars':p4,
    'Capital_Words':p5,
    'Has_Attachment':p6,
    'Sender_Domain':p7
},index=[0])

    st.write("")

    # ---------- Prediction ----------
    if st.button("🔍 Check Email"):
        pred = model.predict(data_new)
        prob = float(model.predict_proba(data_new)[0][1])

        if pred[0] == 1:
            st.markdown('<div class="result-spam">🚨 This email is SPAM</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-ok">✅ This email is NOT spam</div>', unsafe_allow_html=True)

        st.write("")
        c1, c2 = st.columns(2)
        c1.metric("Spam probability", f"{prob*100:.1f}%")
        c2.metric("Not spam probability", f"{(1-prob)*100:.1f}%")
        st.progress(prob)

        with st.expander("See the details you entered"):
            st.write(pd.DataFrame({
                "Feature": ["Subject","Sender domain","Email length","Links","Special characters","Capital words","Attachment"],
                "Value": [s1, s3, p2, p3, p4, p5, s2]
            }).astype(str))

    st.write("")
    st.caption("Made with Streamlit • Email Spam Detection Project")

if __name__ == '__main__':
    main()