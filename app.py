# -*- coding:utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd

def main():

    st.markdown("# Hi")
    st.write(np.__version__)
    st.write(pd.__version__)

if __name__=="__main__":
    main()