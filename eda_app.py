# -*- coding:utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def run_eda_app():

    st.subheader("EDA Page")
    st.subheader("Things going well..")

    iris_df = pd.read_csv("data/iris.csv")

    submenu = st.sidebar.selectbox("Submenu", ["Statistics", "Visualization", "Plots"])
    
    if submenu == "Statistics":
        st.subheader("Statistics")
    elif submenu == "Visualization":
        st.subheader("Visualization")
        fig_1 = px.scatter(
            iris_df,
            x="sepal_width",
            color="species",
            size="petal_length",
            hover_data=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            title="Scatter Plot of Sepal Width vs Petal Length"
        )
        st.plotly_chart(fig_1)
    elif submenu == "Plots":
        st.subheader("Plots")
    else:
        pass

    st.dataframe(iris_df)
