# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2014 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Simple example that connects to the first Crazyflie found, logs the Stabilizer
and prints it to the console. After 10s the application disconnects and exits.
"""
import logging
import time
from threading import Timer

import cflib.crtp  # noqa
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

uri = uri_helper.uri_from_env(default='radio://0/60/2M/E7E7E7E716')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)
SAFE_DISTANCE = 400 #sensor reading
SAFE_DISTANCE_LP = 200 # Sensor reading in lp zone
DEFAULT_HEIGHT = 0.3 #m
DEFAULT_VELOCITY = 0.1 #m/s
SAFE_RIGHT_WALL = 0.3 # meters
SAFE_LEFT_WALL = 0.8 # meters
INITIAL_X = 0.5
INITIAL_Y = 1.8
GOAL_X = 1.5

class LoggingExample:
    """
    Simple logging example class that logs the Stabilizer from a supplied
    link uri and disconnects after 5s.
    """

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """

        self._cf = Crazyflie(rw_cache='./cache')

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)
        self.yaw_direction = True
        print('Connecting to %s' % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

        # State Variables
        self.land = False
        self.lift_off = False

        # Sensor values
        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0
        self.front = 0
        self.back = 0
        self.left = 0
        self.right = 0
        self.iter = 0

        self.default_direction = 'RIGHT'
        self.orientation = 'DEVANT'

        # Search landing pad
        self.direction = "LEFT"
        self.landing_pad_find = False
        self.pass_obstacle = False
        self.x_goback = False
        self.save_x_pos = None
        self.save_y_pos = None

        self.initial_position_reached = False
        self.x_target = GOAL_X

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)

        # The definition of the logconfig can be made before connecting
        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=50)
        self._lg_stab.add_variable('stateEstimate.x', 'float')
        self._lg_stab.add_variable('stateEstimate.y', 'float')
        self._lg_stab.add_variable('stateEstimate.z', 'float')
        self._lg_stab.add_variable('stabilizer.yaw', 'float')
        self._lg_stab.add_variable('range.front')
        self._lg_stab.add_variable('range.back')
        self._lg_stab.add_variable('range.left')
        self._lg_stab.add_variable('range.right')
        
        # The fetch-as argument can be set to FP16 to save space in the log packet
        # self._lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_stab)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

        # Start a timer to disconnect in 10s
        t = Timer(50, self._cf.close_link)
        t.start()

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        self.x = data['stateEstimate.x']
        self.y = data['stateEstimate.y']
        self.z = data['stateEstimate.z']
        self.yaw = data['stabilizer.yaw']
        #print(self.x)
        if np.abs(self.yaw) <= 45:
            self.front = data['range.front']
            self.back = data['range.back']
            self.left = data['range.left']
            self.right = data['range.right']
            self.orientation = 'DEVANT'

        elif self.yaw >45 and self.yaw<=135:
            self.front = data['range.right']
            self.back = data['range.left']
            self.left = data['range.front']
            self.right = data['range.back']
            self.orientation = 'GAUCHE'
            #print('GAUCHE')

        elif self.yaw < -45 and self.yaw > -135:
            self.front = data['range.left']
            self.back = data['range.right']
            self.left = data['range.back']
            self.right = data['range.front']
            self.orientation = 'DROITE'
            #print('DROITE')

        elif 135 < self.yaw < 180 or -180 > self.yaw > -135:
            self.front = data['range.back']
            self.back = data['range.front']
            self.left = data['range.right']
            self.right = data['range.left']
            self.orientation = 'RETOURNE'
            #print('RETOURNE')

        if self.orientation == 'DEVANT':
            if self.yaw >= 0:
                self.default_direction = 'RIGHT'
            else:
                self.default_direction = 'LEFT'
        elif self.orientation == 'GAUCHE':
            if self.yaw >= 90:
                self.default_direction = 'RIGHT'
            elif self.yaw < 90:
                self.default_direction = 'LEFT'
        elif self.orientation == 'RETOURNE':
            if -135 >= self.yaw >= -180:
                self.default_direction = 'RIGHT'
            elif 135 <= self.yaw <= 180:
                self.default_direction = 'LEFT'
        elif self.orientation == 'DROITE':
            if self.yaw <= -90:
                self.default_direction = 'RIGHT'
            else:
                self.default_direction = 'LEFT'


        
    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        self.is_connected = False

    def is_obstacle_close(self):
        sensor_data = [self.back,self.front,self.left,self.right]
        return any(sensor_data) and min(sensor_data) < SAFE_DISTANCE

    def obstacle_avoidance_routine(self):
        """Simple obstacle avoidance routine."""
        if self.front < SAFE_DISTANCE:
            # If obstacle detected in front, move left or right based on left and right sensor readings
            if self.default_direction == 'RIGHT':
                print("Hi")
                print(self.y)
                if self.y > SAFE_LEFT_WALL:
                    # Move right
                    return (0, DEFAULT_VELOCITY), (self.x, self.y - DEFAULT_VELOCITY)
                else:
                    # Move left
                    return (0, DEFAULT_VELOCITY), (self.x, self.y + DEFAULT_VELOCITY)
            else:
                print("Hello")
                print(self.y)
                if self.y < SAFE_RIGHT_WALL:
                    # Move right
                    return (0, DEFAULT_VELOCITY), (self.x, self.y + DEFAULT_VELOCITY)
                else:
                    # Move right
                    return (0, -DEFAULT_VELOCITY), (self.x, self.y - DEFAULT_VELOCITY)
        elif self.left < SAFE_DISTANCE:
            if self.y < SAFE_RIGHT_WALL:
                return (0, 0), (self.x,self.y)
            else:
            # If obstacle detected on the left, move right
                return (0, -DEFAULT_VELOCITY), (self.x, self.y - DEFAULT_VELOCITY)
        elif self.right < SAFE_DISTANCE:
            if self.y > SAFE_LEFT_WALL:
                return (0, 0), (self.x,self.y)
            else:
            # If obstacle detected on the right, move left
                return (0, DEFAULT_VELOCITY), (self.x, self.y + DEFAULT_VELOCITY)
        elif self.back < SAFE_DISTANCE:
            # If obstacle detected at the back, move forward
            return (DEFAULT_VELOCITY, 0), (self.x + DEFAULT_VELOCITY, self.y)
        else:
            # No obstacles detected, maintain current velocity and position
            return (0, 0), (self.x,self.y)
    
    
    def clip_angle(self,angle):
        angle = angle%(360)
        if angle > 180:
            angle -= 360
        if angle < -180:
            angle += 360
        return angle

    def move_command(self):
        # Start by taking off and hovering at the initial position.
        sensor_values = {
            'front': self.front,
            'right': self.right,
            'back': self.back,
            'left': self.left
        }
    
        # Find the direction with the maximum sensor value
        max_direction = max(sensor_values, key=sensor_values.get)
        max_value = sensor_values[max_direction]

        # Print the result
        yaw_goal = self.yaw + 10
        yaw_goal = self.clip_angle(yaw_goal)
        
        if not self.initial_position_reached:           
            if self.x >= self.x_target:
                self.initial_position_reached = True
                print("Reached 3 meters in X")
                self._cf.commander.send_position_setpoint(self.x, self.y, DEFAULT_HEIGHT, yaw_goal)
                return
            else:
                # Check for obstacle proximity
                
                if self.is_obstacle_close():
                    velocity, position = self.obstacle_avoidance_routine()
                    self._cf.commander.send_position_setpoint(position[0], position[1], DEFAULT_HEIGHT,yaw_goal)
                    time.sleep(0.05)
                else: 
                    self._cf.commander.send_position_setpoint(self.x + 0.1, self.y, DEFAULT_HEIGHT, yaw_goal)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.05)  # Adjust sleep time based on responsiveness needs








    def is_obstacle_close_lp(self):
        sensor_data = [self.back,self.front,self.left,self.right]
        if self.direction == "LEFT":
            return sensor_data[2] < SAFE_DISTANCE_LP
        else:
            return sensor_data[3] < SAFE_DISTANCE_LP
        
    def obstacle_avoidance_left(self):
        self.save_x_pos = self.x
        self.save_y_pos = self.y
        while True:
            if not self.pass_obstacle:
                if self.left < SAFE_DISTANCE_LP or (self.x - self.save_x_pos) < 0.1:
                    self._cf.commander.send_position_setpoint(self.x + 0.1, self.y, DEFAULT_HEIGHT, self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                else:
                    self._cf.commander.send_position_setpoint(self.x + 0.1, self.y, DEFAULT_HEIGHT, self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                    self.pass_obstacle = True
            else:
                if not self.x_goback:
                    if self.back < SAFE_DISTANCE_LP or (self.y - self.save_y_pos) < 0.1:
                        if self.y < SAFE_LEFT_WALL:
                            self._cf.commander.send_position_setpoint(self.x, self.y + 0.1, DEFAULT_HEIGHT, self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                            time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                        else:
                            self.direction = "RIGHT"
                            self._cf.commander.send_position_setpoint(self.x, self.y, DEFAULT_HEIGHT, self.y)
                            self.pass_obstacle = False
                            self.x_goback = False
                            break
                    else:
                        self._cf.commander.send_position_setpoint(self.x, self.y + 0.1, DEFAULT_HEIGHT, self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                        time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                        self.x_goback = True
                else:
                    self._cf.commander.send_position_setpoint(self.x - (self.x - self.save_x_pos), self.y+0.1, DEFAULT_HEIGHT, self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.5)  # Adjust sleep time based on responsiveness needs
                    self.pass_obstacle = False
                    self.x_goback = False
                    break

    def obstacle_avoidance_right(self):
        self.save_x_pos = self.x
        self.save_y_pos = self.y
        while True:
            if not self.pass_obstacle:
                if self.right < SAFE_DISTANCE_LP or (self.x - self.save_x_pos) < 0.1:
                    self._cf.commander.send_position_setpoint(self.x + 0.1, self.y, DEFAULT_HEIGHT, self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                else:
                    self._cf.commander.send_position_setpoint(self.x + 0.1, self.y, DEFAULT_HEIGHT, self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                    self.pass_obstacle = True
            else:
                if not self.x_goback:
                    if self.back < SAFE_DISTANCE_LP or (self.save_y_pos - self.y) < 0.1:
                        if self.y < SAFE_LEFT_WALL:
                            self._cf.commander.send_position_setpoint(self.x, self.y - 0.1, DEFAULT_HEIGHT, self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                            time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                        else:
                            self.direction = "LEFT"
                            self._cf.commander.send_position_setpoint(self.x, self.y, DEFAULT_HEIGHT, self.y)
                            self.pass_obstacle = False
                            self.x_goback = False
                            break
                    else:
                        self._cf.commander.send_position_setpoint(self.x, self.y - 0.1, DEFAULT_HEIGHT, self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                        time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                        self.x_goback = True
                else:
                    self._cf.commander.send_position_setpoint(self.x - (self.x - self.save_x_pos), self.y-0.1, DEFAULT_HEIGHT, self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.5)  # Adjust sleep time based on responsiveness needs
                    self.pass_obstacle = False
                    self.x_goback = False
                    break



    def obstacle_avoidance_lp(self):
        """Simple obstacle avoidance routine."""
        if self.left < SAFE_DISTANCE_LP:
            if self.y > SAFE_LEFT_WALL:
                return (0, 0), (self.x,self.y)
            else:
                # If obstacle detected on the left, move right
                self.obstacle_avoidance_left()
        elif self.right < SAFE_DISTANCE_LP:
            if self.y < SAFE_RIGHT_WALL:
                return (0, 0), (self.x,self.y)
            else:
                # If obstacle detected on the left, move right
                self.obstacle_avoidance_right()
            

    def search_landing_pad(self):
        
        if self.x > 3.0:
            print("Reached Landing pad")
            self.land = True
        
        # Start by taking off and hovering at the initial position.
        sensor_values = {
            'front': self.front,
            'right': self.right,
            'back': self.back,
            'left': self.left
        }
    
        # Find the direction with the maximum sensor value
        max_direction = max(sensor_values, key=sensor_values.get)
        max_value = sensor_values[max_direction]

        # Print the result
        yaw_goal = self.yaw + 10
        yaw_goal = self.clip_angle(yaw_goal)
        print(self.y)
        if self.landing_pad_find:           
            pass
        else:
            # Check for obstacle proximity          
            if self.is_obstacle_close_lp():
                self.obstacle_avoidance_lp()
                time.sleep(0.05)
            else:
                if self.direction == "LEFT":
                    if self.y < SAFE_LEFT_WALL + 0.1:
                        self._cf.commander.send_position_setpoint(self.x, self.y + 0.1, DEFAULT_HEIGHT, yaw_goal)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                        time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                    else:
                        self._cf.commander.send_position_setpoint(self.x + 0.2, self.y, DEFAULT_HEIGHT, yaw_goal)
                        self.direction = "RIGHT"
                        time.sleep(0.5)
                else:
                    if self.y > SAFE_RIGHT_WALL - 0.1:
                        self._cf.commander.send_position_setpoint(self.x, self.y - 0.1, DEFAULT_HEIGHT, yaw_goal)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                        time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                    else:
                        self._cf.commander.send_position_setpoint(self.x + 0.2, self.y, DEFAULT_HEIGHT, yaw_goal)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                        self.direction = "LEFT"
                        time.sleep(0.5)  # Adjust sleep time based on responsiveness needs


        


if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    le = LoggingExample(uri)
    cf = le._cf

    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)
    '''
    cf.param.set_value('kalman.initialX', INITIAL_X)
    cf.param.set_value('kalman.initialY', INITIAL_Y)
    cf.param.set_value('kalman.initialZ', 0)
    '''
    
    # The Crazyflie lib doesn't contain anything to keep the application alive,
    # so this is where your application should do something. In our case we
    # are just waiting until we are disconnected.
    try: 
        while le.is_connected:
            time.sleep(0.01)
            
            if le.land:
                for _ in range(20):
                    print('landing')
                    cf.commander.send_hover_setpoint(0, 0, 0, 0.3)
                    time.sleep(0.1)

                for y in range(10):
                    print('landing')
                    cf.commander.send_hover_setpoint(0, 0, 0, (10 - y) / 20)
                    time.sleep(0.1)

                cf.commander.send_stop_setpoint()
                le.is_connected = False
                break

            if not le.lift_off:
                for y in range(10):
                    cf.commander.send_hover_setpoint(0, 0, 0, y / 20)
                    time.sleep(0.1)

                for _ in range(20):
                    cf.commander.send_hover_setpoint(0, 0, 3, 0.3)
                    time.sleep(0.1)
                le.lift_off = True
                print('lift off')
            elif not le.initial_position_reached:
                le.move_command()
                le.iter +=1
            elif le.initial_position_reached:
                le.search_landing_pad()
                
    except KeyboardInterrupt:
        cf.commander.send_stop_setpoint()

        




