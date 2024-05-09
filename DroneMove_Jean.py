import logging
import time
import cflib
import math
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.positioning.position_hl_commander import PositionHlCommander#, MotionCommander
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.commander import Commander
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils.multiranger import Multiranger

# Constants
URI = 'radio://0/60/2M/E7E7E7E716'
DEFAULT_HEIGHT = 0.3       # meters
TAKEOFF_HEIGHT = 1.0        # meters
DEFAULT_VELOCITY = 0.1   # m/s
DEFAULT_AVOIDANCE = 0.1     # m
SAFE_DISTANCE = 0.4         # meters, minimum distance to obstacles
current_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}

# Configure logging
logging.basicConfig(level=logging.ERROR)

def position_callback(timestamp, data, logconf):
    global current_position
    current_position['x'] = data['kalman.stateX']
    current_position['y'] = data['kalman.stateY']
    current_position['z'] = data['kalman.stateZ']
    print(f"Position: x={current_position['x']:.2f}, y={current_position['y']:.2f}, z={current_position['z']:.2f}")

def setup_logger(scf):
    """Set up logger configuration."""
    log_config = LogConfig(name='KalmanPosition', period_in_ms=100)
    log_config.add_variable('kalman.stateX', 'float')
    log_config.add_variable('kalman.stateY', 'float')
    log_config.add_variable('kalman.stateZ', 'float')

    return log_config


def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break


def set_initial_position(scf, x, y, z, yaw_deg):
    scf.cf.param.set_value('kalman.initialX', x)
    scf.cf.param.set_value('kalman.initialY', y)
    scf.cf.param.set_value('kalman.initialZ', z)

    yaw_radians = math.radians(yaw_deg)
    scf.cf.param.set_value('kalman.initialYaw', yaw_radians)



def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    wait_for_position_estimator(cf)

def move_command(scf, mr):
    global current_position
    # PositionHlCommander provides easy-to-use high-level commands.
    with Commander(scf) as pc:
        x = current_position['x']
        y = current_position['y']
        # Start by taking off and hovering at the initial position.
        original_time = time.time()
        while((time.time() - original_time) < 1.5):
            print('send_position_setpoing')
            pc.send_position_setpoint(x, y, 0.5, 0)
        time.sleep(1.)
        initial_position_reached = False

        while not initial_position_reached:
            #print("current position [x y]: ", pc.get_position[0], pc.get_position[1])
            # Check if reached 3 meters in X from the callback or shared state
            
            current_position['x'] = x
            if current_position['x'] >= 2.0:
                initial_position_reached = True
                print("Reached 3 meters in X")
            else:
                # Check for obstacle proximity
                if is_obstacle_close([mr.front, mr.back, mr.left, mr.right]):
                    pc.send_notify_setpoint_stop(remain_valid_milliseconds=50)
                    velocity, position = obstacle_avoidance_routine(mr, current_position, DEFAULT_AVOIDANCE)
                    pc.send_position_setpoint(position[0], position[1], DEFAULT_HEIGHT, 0)  # Avoid obstacles
                    print('just called go to')
                else:
                    print('calling go to')
                    pc.send_notify_setpoint_stop(remain_valid_milliseconds=50)
                    pc.send_position_setpoint(position[0], position[1], DEFAULT_HEIGHT, 0)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    print('just called go to')
            time.sleep(0.5)  # Adjust sleep time based on responsiveness needs

        # Additional commands can be added here based on tasks
        pc.send_position_setpoint(x, y, 0, 0)



def is_obstacle_close(sensor_data):
    return any(sensor_data) and min(sensor_data) < SAFE_DISTANCE

def obstacle_avoidance_routine(mr, position, velocity):
    """Simple obstacle avoidance routine."""
    if mr.front < SAFE_DISTANCE:
        # If obstacle detected in front, move left or right based on left and right sensor readings
        if mr.left < mr.right:
            # Move left
            return (0, velocity), (position['x'], position['y'] + velocity)
        else:
            # Move right
            return (0, -velocity), (position['x'], position['y'] - velocity)
    elif mr.left < SAFE_DISTANCE:
        # If obstacle detected on the left, move right
        return (0, -velocity), (position['x'], position['y'] - velocity)
    elif mr.right < SAFE_DISTANCE:
        # If obstacle detected on the right, move left
        return (0, velocity), (position['x'], position['y'] + velocity)
    elif mr.back < SAFE_DISTANCE:
        # If obstacle detected at the back, move forward
        return (velocity, 0), (position['x'] + velocity, position['y'])
    else:
        # No obstacles detected, maintain current velocity and position
        return (0, 0), position


def main():
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    # Connect to the Crazyflie
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        # on the floor
        initial_x = 1.0
        initial_y = 1.0
        initial_z = 0.0
        initial_yaw = 90 # In degrees / 0: positive X direction / 90: positive Y direction / 180: negative X direction / 270: negative Y direction
        # Initialize Multiranger
        #call sensors 
        log_config = setup_logger(scf)
        scf.cf.log.add_config(log_config)
        log_config.data_received_cb.add_callback(position_callback)
        log_config.start()

        mr = Multiranger(scf)

        reset_estimator(scf)
        #init kalman
        set_initial_position(scf, initial_x, initial_y, initial_z, initial_yaw)
        time.sleep(1.)
        try:
            move_command(scf, mr)
        except:
            pass


if __name__ == '__main__':
    main()

