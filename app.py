# -*- coding:utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd

from eda_app import run_eda_app

from ml_app import run_ml_app

def main():

    st.markdown("# Hi")
    st.write(np.__version__)
    st.write(pd.__version__)

    menu = ["Home", "EDA", "ML", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
    elif choice == "EDA":
        run_eda_app()
    elif choice == "ML":
        run_ml_app()
    elif choice == "About":
        st.subheader("About")
    else:
        pass

if __name__=="__main__":
    main()