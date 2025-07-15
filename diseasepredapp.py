import streamlit as st
import numpy as np
import joblib

liver_model = joblib.load(r"E:\GUVI\Project 4\liver_xgb_model.pkl")
liver_scaler = joblib.load(r"C:\Users\Bala viknese\PycharmProjects\Pythonproject4\scaler.pkl")

kidney_model = joblib.load(r"E:\GUVI\Project 4\kidney_rf_model.pkl")
kidney_scaler = joblib.load(r"C:\Users\Bala viknese\PycharmProjects\Pythonproject4\kidney_scaler.pkl")

parkinsons_model = joblib.load(r"E:\GUVI\Project 4\parkinsons_rf_model.pkl")
parkinsons_scaler = joblib.load(r"C:\Users\Bala viknese\PycharmProjects\Pythonproject4\parkinsons_scaler.pk")

st.title("🧬 Multiple Disease Prediction App")

tab1, tab2, tab3 = st.tabs(["Liver Disease", "Kidney Disease", "Parkinson's Disease"])

# ------------------ Liver Disease Tab ------------------
with tab1:
    st.header("🧬 Liver Disease Prediction")
    age = st.number_input('Age', min_value=1, max_value=120, value=45)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    total_bilirubin = st.number_input('Total Bilirubin', value=1.0)
    direct_bilirubin = st.number_input('Direct Bilirubin', value=0.5)
    alkphos = st.number_input('Alkaline Phosphotase', value=200)
    sgpt = st.number_input('Alamine Aminotransferase (SGPT)', value=30)
    sgot = st.number_input('Aspartate Aminotransferase (SGOT)', value=35)
    total_proteins = st.number_input('Total Proteins', value=6.5)
    albumin = st.number_input('Albumin', value=3.0)
    a_g_ratio = st.number_input('Albumin and Globulin Ratio', value=1.0)

    gender_encoded = 1 if gender == 'Male' else 0

    if st.button('Predict Liver Disease'):
        liver_data = np.array([[age, gender_encoded, total_bilirubin, direct_bilirubin,
                                alkphos, sgpt, sgot, total_proteins, albumin, a_g_ratio]])
        liver_data_scaled = liver_scaler.transform(liver_data)
        prediction = liver_model.predict(liver_data_scaled)[0]
        prob = liver_model.predict_proba(liver_data_scaled)[0][1]
        if prediction == 1:
            st.error(f"⚠ Patient likely has liver disease. (Risk score: {prob:.2f})")
        else:
            st.success(f"✅ Patient unlikely to have liver disease. (Risk score: {prob:.2f})")

# ------------------ Kidney Disease Tab ------------------
with tab2:
    st.header("🩺 Kidney Disease Prediction")
    age = st.number_input('Age ', min_value=1, max_value=120, value=45, key='kidney_age')
    bp = st.number_input('Blood Pressure', value=80)
    sg = st.number_input('Specific Gravity (e.g., 1.01)', value=1.02)
    al = st.number_input('Albumin', value=1.0)
    su = st.number_input('Sugar', value=0.0)
    rbc = st.selectbox('Red Blood Cells', ['normal', 'abnormal'])
    pc = st.selectbox('Pus Cell', ['normal', 'abnormal'])
    pcc = st.selectbox('Pus Cell Clumps', ['present', 'notpresent'])
    ba = st.selectbox('Bacteria', ['present', 'notpresent'])
    bgr = st.number_input('Blood Glucose Random', value=121.0)
    bu = st.number_input('Blood Urea', value=36.0)
    sc = st.number_input('Serum Creatinine', value=1.2)
    sod = st.number_input('Sodium', value=138.0)
    pot = st.number_input('Potassium', value=4.4)
    hemo = st.number_input('Hemoglobin', value=15.4)
    pcv = st.number_input('Packed Cell Volume', value=44.0)
    wc = st.number_input('White Blood Cell Count', value=7800.0)
    rc = st.number_input('Red Blood Cell Count', value=5.2)
    htn = st.selectbox('Hypertension', ['yes', 'no'])
    dm = st.selectbox('Diabetes Mellitus', ['yes', 'no'])
    cad = st.selectbox('Coronary Artery Disease', ['yes', 'no'])
    appet = st.selectbox('Appetite', ['good', 'poor'])
    pe = st.selectbox('Pedal Edema', ['yes', 'no'])
    ane = st.selectbox('Anemia', ['yes', 'no'])

    binary_map = {'yes': 1, 'no': 0, 'normal': 1, 'abnormal': 0,
                  'present': 1, 'notpresent': 0, 'good': 1, 'poor': 0}

    kidney_data = np.array([[age, bp, sg, al, su,
                             binary_map[rbc], binary_map[pc], binary_map[pcc], binary_map[ba],
                             bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
                             binary_map[htn], binary_map[dm], binary_map[cad], binary_map[appet],
                             binary_map[pe], binary_map[ane]]])

    if st.button('Predict Kidney Disease'):
        kidney_data_scaled = kidney_scaler.transform(kidney_data)
        prediction = kidney_model.predict(kidney_data_scaled)[0]
        prob = kidney_model.predict_proba(kidney_data_scaled)[0][1]
        if prediction == 1:
            st.error(f"⚠ High risk of CKD. (Risk score: {prob:.2f})")
        else:
            st.success(f"✅ Low risk of CKD. (Risk score: {prob:.2f})")

# ------------------ Parkinson's Disease Tab ------------------
with tab3:
    st.header("🧠 Parkinson's Disease Prediction")
    st.markdown("Enter patient voice measurements below:")

    mdvp_fo = st.number_input('MDVP:Fo(Hz)', value=119.992)
    mdvp_fhi = st.number_input('MDVP:Fhi(Hz)', value=157.302)
    mdvp_flo = st.number_input('MDVP:Flo(Hz)', value=74.997)
    mdvp_jitter_percent = st.number_input('MDVP:Jitter(%)', value=0.00784)
    mdvp_jitter_abs = st.number_input('MDVP:Jitter(Abs)', value=0.00007)
    mdvp_rap = st.number_input('MDVP:RAP', value=0.0037)
    mdvp_ppq = st.number_input('MDVP:PPQ', value=0.00554)
    jitter_ddp = st.number_input('Jitter:DDP', value=0.01109)
    mdvp_shimmer = st.number_input('MDVP:Shimmer', value=0.04374)
    mdvp_shimmer_db = st.number_input('MDVP:Shimmer(dB)', value=0.426)
    shimmer_apq3 = st.number_input('Shimmer:APQ3', value=0.02182)
    shimmer_apq5 = st.number_input('Shimmer:APQ5', value=0.0313)
    mdvp_apq = st.number_input('MDVP:APQ', value=0.02971)
    shimmer_dda = st.number_input('Shimmer:DDA', value=0.06545)
    nhr = st.number_input('NHR', value=0.02211)
    hnr = st.number_input('HNR', value=21.033)
    rpde = st.number_input('RPDE', value=0.414783)
    dfa = st.number_input('DFA', value=0.815285)
    spread1 = st.number_input('spread1', value=-4.813031)
    spread2 = st.number_input('spread2', value=0.266482)
    d2 = st.number_input('D2', value=2.301442)
    ppe = st.number_input('PPE', value=0.284654)

    parkinsons_data = np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs,
                                 mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db,
                                 shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr,
                                 rpde, dfa, spread1, spread2, d2, ppe]])

    if st.button("Predict Parkinson's Disease"):
        parkinsons_scaled = parkinsons_scaler.transform(parkinsons_data)
        prediction = parkinsons_model.predict(parkinsons_scaled)[0]
        prob = parkinsons_model.predict_proba(parkinsons_scaled)[0][1]
        if prediction == 1:
            st.error(f"⚠ Likely Parkinson's Disease. (Risk score: {prob:.2f})")
        else:
            st.success(f"✅ Unlikely to have Parkinson's Disease. (Risk score: {prob:.2f})")
