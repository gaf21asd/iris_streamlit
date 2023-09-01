# -*- coding:utf-8 -*-

import streamlit as st
import joblib
import os
import numpy as np

def run_ml_app():
    st.subheader("ML")

    # Layout
    col_1, col_2 = st.columns(2)
    with col_1:
        st.subheader("Enter the number.")
        sepal_length = st.select_slider("Sepal Length", options=np.arange(1, 11))
        sepal_width = st.select_slider("Sepal Width", options=np.arange(1, 11))
        petal_length = st.select_slider("Petal Length", options=np.arange(1, 11))
        petal_width = st.select_slider("Petal Width", options=np.arange(1, 11))

        sample_list = [sepal_length, sepal_width, petal_length, petal_width]
        st.write(sample_list)

    with col_2:
        st.subheader("Check the results of the model.")


        # Updating the model
        model_file = "models/lgr_model_iris0901.pkl"
        model = joblib.load(open(os.path.join(model_file),"rb"))
        st.write(model)

        # Make it arrays
        one_sample = np.array(sample_list).reshape(1,-1)
        st.write(one_sample)
        st.write(one_sample.shape)

        # Predicting the category
        prediction = model.predict(one_sample)
        st.write(prediction)

        # Predicting the probability
        pred_prob = model.predict_proba(one_sample)
        st.write(pred_prob)

        if prediction == 0:
            st.success("Setosa species")
            pred_proba_scores = {
                "Prob 1": pred_prob[0][0] * 100, 
                "Prob 0": pred_prob[0][1] * 100, 
            }
            st.write(pred_proba_scores)
            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/220px-Irissetosa1.jpg')
        elif prediction == 1:
            st.success("[ ] species")
            pred_proba_scores = {
                "Prob 1": pred_prob[0][0] * 100, 
                "Prob 0": pred_prob[0][1] * 100, 
            }
            st.write(pred_proba_scores)
            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Blue_Flag%2C_Ottawa.jpg/220px-Blue_Flag%2C_Ottawa.jpg')
        else:
            st.warning("I don't know!")



