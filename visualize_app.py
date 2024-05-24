import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def init_app(reset=False):
    if 'file_name' not in st.session_state:
        st.session_state.file_name = 'logged_data.csv'

    if 'data' not in st.session_state:
        st.session_state.data = pd.read_csv(st.session_state.file_name, index_col=0)


def build_app():
    st.set_page_config(page_title='Crazyflie Data Visualization', layout='wide')

    st.title('Crazyflie Data Visualization')

    st.text_input('Enter the file name:', value=st.session_state.file_name, key='file_name',
                  on_change=lambda: init_app(reset=True))
    init_app()

    if 'data' in st.session_state:
        data = st.session_state.data

        # Choose fsm states
        states = st.multiselect('Select FSM states:', data['fsm'].unique(), default=data['fsm'].unique())
        data = data[data['fsm'].isin(states)]

        # Select data range
        start, stop = st.slider('Select data range:', 0, len(st.session_state.data), (0, len(st.session_state.data)))
        data = data.iloc[start:stop]

        # Select data columns
        columns = st.multiselect('Select data columns:', data.columns, default=data.columns[:3])
        data = data[columns]

        # Plot data
        if st.toggle('Use interactive plot'):
            pd.options.plotting.backend = 'plotly'
            st.plotly_chart(data)
        else:
            pd.options.plotting.backend = 'matplotlib'
            st.pyplot(data.plot())

        st.divider()

        # Display data
        if st.checkbox('Show data'):
            st.dataframe(data)




