# CrazyFly hardware project

## Code base
- FSM: handles the transition between state **fsm_update()**
handles all the required logic
- Default starting position is (0, 0), so need to adapt the map in accordance
- Data can be logged with **log_data()**, variable name must be added in the Dataframe
at the end of the file
- Data can be printed during the run according to the FSM state
with **print_state_info()**

## Install
Install the required dependencies
`pip install requirements.txt`

## Run
To run the main controller

`python main.py` : no obstacle avoidance and no grid search (seb et enrique)

`python main_merged.py` : everything together

## View data logged with streamlit
`streamlit run visualize_app.py`

