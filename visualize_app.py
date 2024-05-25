import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def init_app(reset=False):
    if 'file_name' not in st.session_state:
        st.session_state.file_name = 'logged_data.csv'

    if 'data' not in st.session_state:
        st.session_state.data = pd.read_csv(st.session_state.file_name, index_col=0)


def build_app():
    st.set_page_config(page_title='Crazyflie Data Visualization', layout='wide')

    st.title('Crazyflie Data Visualization')

    init_app()
    st.text_input('Enter the file name:', value=st.session_state.file_name, key='file_name', on_change=lambda: init_app(reset=True))
    init_app()

    if 'data' in st.session_state:
        data = st.session_state.data

        # Choose fsm states
        states = st.multiselect('Select FSM states:', data['fsm'].unique(), default=data['fsm'].unique())
        data = data[data['fsm'].isin(states)]

        # Select data range
        start, stop = st.slider('Select data range:', 0, len(st.session_state.data), (0, len(st.session_state.data)))
        data = data.iloc[start:stop]

        # Add variables

        # Scale range values
        data['down_scaled'] = data['down'].values / 1000
        data['down_diff'] = data['down'].diff() / 100

        diff = st.number_input('Enter the scaling factor:', value=3, key='az diff')
        data['az_diff'] = data['az'].diff(diff)


        # Detect fsm transitions
        scaling = st.number_input('Enter the transition scaling:', value=2.0, key='scaling')
        data['fsm_change'] = data['fsm'].ne(data['fsm'].shift()).astype(int) * scaling

        # Select data columns
        st.text(f'Select data columns to plot: {data.columns}')
        columns = st.multiselect(label='Select data columns:', options=list(data.columns), default=['x', 'y', 'z'])
        data = data[columns]

        # Plot data
        if st.toggle('Use interactive plot'):
            pd.options.plotting.backend = "plotly"
            plotly_fig = go.Figure()
            res = data.plot()
            st.plotly_chart(res, use_container_width=True)
        else:
            pd.options.plotting.backend = 'matplotlib'
            fig, ax = plt.subplots(figsize=(15, 6))
            data.plot(ax=ax)
            st.pyplot(fig, use_container_width=True)
        st.divider()

        # Display data
        if st.checkbox('Show data'):
            st.dataframe(data)

if __name__ == '__main__':
    build_app()