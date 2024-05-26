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
from collections import deque
from threading import Timer
from enum import Enum
import pandas as pd
import cflib.crtp  # noqa
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from matplotlib import pyplot as plt
from spiral import archimedean_spiral, normalize_spiral, circle, star
from a_star_algo import a_star_search
uri = uri_helper.uri_from_env(default='radio://0/60/2M/E7E7E7E716')
from rdp import rdp

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

# Initial Position
INITIAL_X = 0.4
INITIAL_Y = 1.9
LANDING_PAD_SIZE = 0.30

# General Navigation
DEFAULT_HEIGHT = 0.3  #m
DEFAULT_VELOCITY = 0.4  #m/s
GOAL_X = 3.5  # Landing zone
RETURN_GOAL_X = 1.5  # Take off zone
AZ_THRESHOLD = 0.08  #m/s^2  Detecting landing pad
ORIGIN_THRESHOLD = 0.1  #m   Start spiraling

# Obstacle avoidance params
SAFE_DISTANCE = 400  #sensor reading
SAFE_DISTANCE_LP = 200 # Sensor reading in lp zone
SAFE_RIGHT_WALL = 0.3 - INITIAL_Y
SAFE_LEFT_WALL = 2.7 - INITIAL_Y

# Occupancy grid parameters
range_max = 2.  # meters
res_pos = 0.15# meters
conf = 0.1
min_x, max_x = 0 , 5   # x valid map values
min_y, max_y = 0 , 3  # y valid map values

limit_right = 0.4
limit_left = 2.7

class FSM(Enum):
    INIT = 'init'
    TAKE_OFF = 'take_off'
    ROTATE = 'rotate'
    CROSS = 'cross'
    SEARCH = 'search'
    FOUND = 'found'
    CENTERING = 'centering'
    LANDING = 'landing'
    STOP = 'stop'
    GOING_BACK = 'going_back'
    SPIRALING = 'spiraling'


class LoggingExample:
    """
    Simple logging example class that logs the Stabilizer from a supplied
    link uri and disconnects after 5s.
    """

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """
        # Sensor values
        self.spiral_iter = 0
        # self.center_iter = 0
        self.take_off_iter = 0
        self.pattern_iter = 0
        self.finding_iter = 0
        self.path_iter = 0
        self.path_iter_first = 0

        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.front = 0
        self.back = 0
        self.left = 0
        self.right = 0
        self.up = 0
        self.down = 0
        self.down_range = 0
        self.down_buffer = deque(maxlen=10)
        self.z_ref = 0.2
        self.az = 0

        self.down_buffer_data = []
        self.fsm = FSM.INIT

        # Data for FSM
        self.found_pos = None  # [x, y, z, yaw]
        self.pattern_position = np.array([])  # [[x, y], ...]
        self.centering_observations = []  # [[x, y, down], ...]
        self.landing_pos = None  # [x, y, z, yaw]
        self.landing_pad_reached = False
        self.spiral_coord = normalize_spiral(archimedean_spiral(num_points=500), fixed_norm=0.02)

        self.t = 0  # Iteration time

        self.occ_map = np.ones((int(5 / res_pos), int(3 / res_pos)))
        self.circle_points = normalize_spiral(circle(radius=LANDING_PAD_SIZE / 1.2), fixed_norm=0.01)
        self.path = None
        # Obstacle avoidance data
        self.default_direction = 'RIGHT'
        self.orientation = 'DEVANT'

        # Search landing pad
        self.direction = "LEFT"
        self.pass_obstacle = False
        self.x_goback = False
        self.save_x_pos = None
        self.save_y_pos = None

        self.path_looking = np.array([[3.6,limit_left],[3.6,limit_right ],[3.9,limit_right ],
                        [3.9,limit_left ],[4.2,limit_left ],[4.2,limit_right ],
                        [4.5,limit_right ],[4.5,limit_left ],[4.7,limit_left ],
                        [4.7,limit_right ]]) - np.array([INITIAL_X, INITIAL_Y])
        self.path_looking_index = 0

        self.actions: dict[FSM, callable] = {
            FSM.INIT: lambda x: time.sleep(0.05),
            FSM.TAKE_OFF: self.take_off,
            FSM.CROSS: self.move_command,  
            FSM.SEARCH: self.search_landing_pad,
            FSM.FOUND: self.finding,
            FSM.CENTERING: self.centering,
            FSM.LANDING: self.landing,
            FSM.STOP: self.stop,
            FSM.GOING_BACK: self.going_back,
            FSM.SPIRALING: self.spiraling,
        }

        self._cf = Crazyflie(rw_cache='./cache')

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % link_uri)
        self._cf.open_link(link_uri)  # Try to connect to the Crazyflie
        self.is_connected = True  # Variable used to keep main loop occupied until disconnect

        self.logs = []

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)

        # The definition of the logconfig can be made before connecting
        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=50)
        self._lg_stab.add_variable('stateEstimate.x', 'FP16')
        self._lg_stab.add_variable('stateEstimate.y', 'FP16')
        self._lg_stab.add_variable('stateEstimate.z', 'FP16')
        self._lg_stab.add_variable('stabilizer.yaw', 'FP16')
        self._lg_stab.add_variable('range.front')
        self._lg_stab.add_variable('range.back')
        self._lg_stab.add_variable('range.left')
        self._lg_stab.add_variable('range.right')
        self._lg_stab.add_variable('range.up')
        # self._lg_stab.add_variable('range.zrange')

        self._lg_stab2 = LogConfig(name='Velocities', period_in_ms=50)
        self._lg_stab2.add_variable('stateEstimate.vx', 'FP16')
        self._lg_stab2.add_variable('stateEstimate.vy', 'FP16')
        self._lg_stab2.add_variable('stateEstimate.vz', 'FP16')
        self._lg_stab2.add_variable('range.zrange', 'FP16')
        self._lg_stab2.add_variable('stateEstimate.az', 'FP16')
        # The fetch-as argument can be set to FP16 to save space in the log packet
        # self._lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_stab)
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            self._lg_stab.start()

            self._cf.log.add_config(self._lg_stab2)
            self._lg_stab2.data_received_cb.add_callback(self._stab_log_data2)
            self._lg_stab2.error_cb.add_callback(self._stab_log_error)
            self._lg_stab2.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

        # Start a timer to disconnect in 10s
        t = Timer(100, self._cf.close_link)
        t.start()

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data2(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        # if self.t % 1 == -1:
        #     print(f'[{timestamp}][{logconf.name}]: ', end='')
        #     for name, value in data.items():
        #         print(f'{name}: {value:3.3f} ', end='')
        #     print()

        self.vx = data['stateEstimate.vx']
        self.vy = data['stateEstimate.vy']
        self.vz = data['stateEstimate.vz']
        self.down = data['range.zrange']
        self.az = data['stateEstimate.az']
        self.down_buffer.append(self.down)

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        # if self.t % 1 == -1:  # Print every 30th iteration
        #     print(f'[{timestamp}][{logconf.name}]: ', end='')
        #     for name, value in data.items():
        #         print(f'{name}: {value:3.3f} ', end='')
        #     print()

        # Update sensor values
        self.x = data['stateEstimate.x']
        self.y = data['stateEstimate.y']
        self.z = data['stateEstimate.z']
        self.yaw = data['stabilizer.yaw']
        self.back = min(data['range.back'], 1000*range_max)
        self.left = min(data['range.left'], 1000*range_max)
        self.front = min(data['range.front'], 1000*range_max)
        self.right = min(data['range.right'], 1000*range_max)
        self.up = data['range.up']

        self.occupancy_map()

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

    def fsm_update(self):
        if self.up < 100 and self.fsm != FSM.LANDING:
            self.landing_pad_reached = True
            self.landing_pos = [self.x, self.y, self.z, self.yaw]
            self.fsm = FSM.LANDING
            return

        if self.fsm == FSM.INIT:
            self.fsm = FSM.TAKE_OFF

        elif self.fsm == FSM.TAKE_OFF:
            self.take_off_iter += 1
            if self.take_off_iter > 20 and self.z < 0.05:
                self.fsm = FSM.STOP
                print('Critical ERROR: Take off failed')
            if self.z > DEFAULT_HEIGHT - 0.05:
                if self.landing_pad_reached:
                    self.fsm = FSM.GOING_BACK
                    self.path = None
                else: 
                    self.fsm = FSM.CROSS
                self.take_off_iter = 0

        elif self.fsm == FSM.CROSS:
            if self.x + INITIAL_X > GOAL_X:
                self.fsm = FSM.SEARCH
                self.path = None
                self.path_iter_first = 0

        elif self.fsm == FSM.SEARCH or self.fsm == FSM.SPIRALING or self.fsm == FSM.GOING_BACK :
            if self.az > AZ_THRESHOLD:
                if self.fsm != FSM.GOING_BACK or self.x < RETURN_GOAL_X:
                    self.found_pos = [self.x, self.y, self.z, self.yaw]
                    yaw = np.arctan2(self.vy, self.vx)
                    rotation_matrix = np.array([
                        [np.cos(yaw), np.sin(yaw)],
                        [-np.sin(yaw), np.cos(yaw)]
                    ])
                    pattern_pos = np.dot(self.circle_points, rotation_matrix)
                    self.pattern_position = pattern_pos + self.found_pos[:2]
                    self.found_vel = [self.vx, self.vy]
                    self.prev_square_pos = self.found_pos[:2]
                    self.fsm = FSM.FOUND
                    print(f'Found: {self.x:.3f}, {self.y:.3f}, {int(self.down)}')
                    print(f'Velocity: {self.vx:.3f}, {self.vy:.3f}')
                    print(f'Yaw: {np.arctan2(self.vy, self.vx):.3f}')

            elif self.fsm == FSM.GOING_BACK:
                if np.linalg.norm([self.x, self.y]) < ORIGIN_THRESHOLD:
                    self.fsm = FSM.SPIRALING

        elif self.fsm == FSM.FOUND:
            if self.finding_iter == 5:
                self.fsm = FSM.CENTERING
                self.finding_iter = 0

        elif self.fsm == FSM.CENTERING:
            if self.pattern_iter == len(self.pattern_position):
                self.landing_pos = self.compute_landing_pos()
                self.centering_observations = []  # Reset centering observations for next landing
                self.pattern_iter = 0
                self.fsm = FSM.LANDING

        elif self.fsm == FSM.LANDING:
            self.fsm = FSM.STOP if self.landing_pad_reached else FSM.TAKE_OFF
            self.landing_pad_reached = True
                
        else:
            print("Invalid state")

    def take_off(self):
        # TODO: Maybe implement take off routine as a send_position_setpoint ???
        # cf.commander.send_position_setpoint(0, 0, DEFAULT_HEIGHT, 0)
        self._cf.commander.send_hover_setpoint(0, 0, 0, DEFAULT_HEIGHT)
        time.sleep(0.05)

    # def rotate(self):
    #     cf.commander.send_hover_setpoint(0, 0, 30, DEFAULT_HEIGHT)
    #     time.sleep(0.05)
    def distance_2d(self,x1,y1,x2,y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def going_back(self):
        yaw_goal = np.sin(self.t*np.pi/20)
        yaw_goal = 20*yaw_goal
        yaw_goal = self.clip_angle(yaw_goal)

        if self.path is None or self.t%20==0:
            
            start_node = [int((self.x +INITIAL_X )/res_pos +0.5) ,
                          int((self.y + INITIAL_Y)/res_pos + 0.5)]
            end_node = [int(( 0+ INITIAL_X)/res_pos + 0.5),
                        int((0 +INITIAL_Y)/res_pos + 0.5)]

            path_test = a_star_search(np.array(self.occ_map), start_node, end_node,self)
            if path_test is not None:
                self.path = np.array(path_test) * res_pos
                self.path -= np.array([INITIAL_X, INITIAL_Y])
            
                self.path_iter_first = 0

        if self.path is not None:
            dist = self.distance_2d(self.x,self.y,self.path[self.path_iter_first][0],self.path[self.path_iter_first][1])
            if (dist<0.06) and self.path_iter_first <= len(self.path)-2:
                self.path_iter_first +=1
            self._cf.commander.send_position_setpoint(self.path[self.path_iter_first][0] , self.path[self.path_iter_first][1], DEFAULT_HEIGHT, yaw_goal)
            time.sleep(0.05)
        else:
            self._cf.commander.send_position_setpoint(self.x, self.y, DEFAULT_HEIGHT, yaw_goal)
            time.sleep(0.05)

    def spiraling(self):
        x, y = self.spiral_coord[self.spiral_iter]
        self.spiral_iter += 1
        if self.spiral_iter == len(self.spiral_coord):
            self.landing_pos = [x, y, DEFAULT_HEIGHT, 0]
            self.fsm = FSM.LANDING
            print("Spiral finished----------------------------")
        self._cf.commander.send_position_setpoint(x, y, DEFAULT_HEIGHT, 0)
        time.sleep(0.05)

    def finding(self):
        x, y = self.pattern_position[0]
        _, _, _, yaw = self.found_pos
        self._cf.commander.send_position_setpoint(x, y, DEFAULT_HEIGHT, yaw)
        self.finding_iter += 1
        time.sleep(0.05)

    def centering(self):
        next_pos = self.pattern_position[self.pattern_iter]
        x, y, _, yaw = self.found_pos
        self.centering_observations.append([self.x, self.y, self.down, self.az])
        self.down_buffer_data.append([d for d in self.down_buffer])
        # z_go = self.z * 0.99 + (DEFAULT_HEIGHT + 0.05) * 0.01
        self._cf.commander.send_position_setpoint(next_pos[0], next_pos[1], DEFAULT_HEIGHT, yaw)
        self.pattern_iter += 1
        time.sleep(0.05)


    def landing(self):
        x, y, _, yaw = self.landing_pos

        for i in range(10):
            self._cf.commander.send_position_setpoint(x, y, DEFAULT_HEIGHT, yaw)
            time.sleep(0.05)

        iter_num = 30
        for i in range(iter_num + 20):
            self._cf.commander.send_position_setpoint(x, y, DEFAULT_HEIGHT - (DEFAULT_HEIGHT / iter_num) * i, yaw)
            if i > iter_num:
                self._cf.commander.send_stop_setpoint()
                # print('Landing')
                self.print_state_info()
            time.sleep(0.1)

    def stop(self):
        self._cf.commander.send_stop_setpoint()
        self._cf.close_link()
        time.sleep(0.05)

    def compute_landing_pos(self):
        observations = np.array(self.centering_observations)
        obs_delta = observations[1:, 2] - observations[:-1, 2]
        rising_edge = np.where(obs_delta > 17)[0]  # Going up on the pad
        falling_edge = np.where(obs_delta < -17)[0]  # Falling off the pad
        mask = np.full(len(observations), np.nan)
        mask[rising_edge] = True
        mask[falling_edge] = False
        current = True
        for i in range(len(mask)):
            if np.isnan(mask[i]):
                mask[i] = current
            else:
                current = mask[i]
        mask = mask.astype(bool)
        print("Mask")
        print(mask)
        masked_obs = observations[mask]
        land_pos = np.mean(masked_obs[:, :2], axis=0)
        return [land_pos[0], land_pos[1], self.found_pos[2], self.found_pos[3]]

    def is_obstacle_close(self):
        sensor_data = [self.back, self.front, self.left, self.right]
        return any(sensor_data) and min(sensor_data) < SAFE_DISTANCE

    def obstacle_avoidance_routine(self):
        """Simple obstacle avoidance routine."""
        if self.front < SAFE_DISTANCE:
            # If obstacle detected in front, move left or right based on left and right sensor readings
            if self.default_direction == 'RIGHT':
                if self.y > SAFE_LEFT_WALL:
                    # Move right
                    return (0, DEFAULT_VELOCITY), (self.x, self.y - DEFAULT_VELOCITY)
                else:
                    # Move left
                    return (0, DEFAULT_VELOCITY), (self.x, self.y + DEFAULT_VELOCITY)
            else:
                
                print(self.y)
                if self.y < SAFE_RIGHT_WALL:
                    # Move right
                    return (0, DEFAULT_VELOCITY), (self.x, self.y + DEFAULT_VELOCITY)
                else:
                    # Move right
                    return (0, -DEFAULT_VELOCITY), (self.x, self.y - DEFAULT_VELOCITY)
        elif self.left < SAFE_DISTANCE:
            if self.y < SAFE_RIGHT_WALL:
                return (0, 0), (self.x, self.y)
            else:
                # If obstacle detected on the left, move right
                return (0, -DEFAULT_VELOCITY), (self.x, self.y - DEFAULT_VELOCITY)
        elif self.right < SAFE_DISTANCE:
            if self.y > SAFE_LEFT_WALL:
                return (0, 0), (self.x, self.y)
            else:
                # If obstacle detected on the right, move left
                return (0, DEFAULT_VELOCITY), (self.x, self.y + DEFAULT_VELOCITY)
        elif self.back < SAFE_DISTANCE:
            # If obstacle detected at the back, move forward
            return (DEFAULT_VELOCITY, 0), (self.x + DEFAULT_VELOCITY, self.y)
        else:
            # No obstacles detected, maintain current velocity and position
            return (0, 0), (self.x, self.y)

    def clip_angle(self, angle):
        angle = angle % (360)
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
        
        yaw_goal = np.sin(self.t*np.pi/20)
        yaw_goal = 20*yaw_goal
        yaw_goal = self.clip_angle(yaw_goal)
        

        if self.t%20 == 0:
            start_node = [int((self.x +INITIAL_X )/res_pos +0.5) ,int((self.y + INITIAL_Y)/res_pos + 0.5)]
            end_node = [int((self.x +INITIAL_X + 2)/res_pos + 0.5),int((self.y +INITIAL_Y)/res_pos + 0.5)]
            path_test = a_star_search(np.array(self.occ_map), start_node, end_node,self)
            
            if path_test is not None:
                self.path = np.array(path_test) * res_pos
                self.path -= np.array([INITIAL_X, INITIAL_Y])
            
                """  if self.path is not None and (np.any(self.path)):
                
                #rdp algorithm to simplify the path
                self.path = rdp(self.path,epsilon=1)
                self.path[0] = [self.x,self.y]
                self.path = normalize_spiral(np.array(self.path),fixed_norm=0.01)  """
                self.path_iter_first = 0
            

        if self.path is not None:
            dist = self.distance_2d(self.x,self.y,self.path[self.path_iter_first][0],self.path[self.path_iter_first][1])
            
            
            if (dist<0.06) and self.path_iter_first <= len(self.path)-2:
                self.path_iter_first +=1
            self._cf.commander.send_position_setpoint(self.path[self.path_iter_first][0] , self.path[self.path_iter_first][1], DEFAULT_HEIGHT, yaw_goal)
            time.sleep(0.05)
        else:
            self._cf.commander.send_position_setpoint(self.x, self.y, DEFAULT_HEIGHT, yaw_goal)
            time.sleep(0.05)
            

        """ if self.is_obstacle_close():
            velocity, position = self.obstacle_avoidance_routine()
            self._cf.commander.send_position_setpoint(position[0], position[1], DEFAULT_HEIGHT, yaw_goal)
            time.sleep(0.05)
        else:
            self._cf.commander.send_position_setpoint(self.x + 0.1, self.y, DEFAULT_HEIGHT, yaw_goal)  # Target x=3 meters, y=0 (no change), z=0.5 meters
            time.sleep(0.05)  # Adjust sleep time based on responsiveness needs """

    def is_obstacle_close_lp(self):
        sensor_data = [self.back, self.front, self.left, self.right]
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
                    self._cf.commander.send_position_setpoint(self.x + 0.1, self.y, DEFAULT_HEIGHT,
                                                              self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                else:
                    self._cf.commander.send_position_setpoint(self.x + 0.1, self.y, DEFAULT_HEIGHT,
                                                              self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                    self.pass_obstacle = True
            else:
                if not self.x_goback:
                    if self.back < SAFE_DISTANCE_LP or (self.y - self.save_y_pos) < 0.1:
                        if self.y < SAFE_LEFT_WALL:
                            self._cf.commander.send_position_setpoint(self.x, self.y + 0.1, DEFAULT_HEIGHT,
                                                                      self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                            time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                        else:
                            self.direction = "RIGHT"
                            self._cf.commander.send_position_setpoint(self.x, self.y, DEFAULT_HEIGHT, self.y)
                            self.pass_obstacle = False
                            self.x_goback = False
                            break
                    else:
                        self._cf.commander.send_position_setpoint(self.x, self.y + 0.1, DEFAULT_HEIGHT,
                                                                  self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                        time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                        self.x_goback = True
                else:
                    self._cf.commander.send_position_setpoint(self.x - (self.x - self.save_x_pos), self.y + 0.1,
                                                              DEFAULT_HEIGHT,
                                                              self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
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
                    self._cf.commander.send_position_setpoint(self.x + 0.1, self.y, DEFAULT_HEIGHT,
                                                              self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                else:
                    self._cf.commander.send_position_setpoint(self.x + 0.1, self.y, DEFAULT_HEIGHT,
                                                              self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                    self.pass_obstacle = True
            else:
                if not self.x_goback:
                    if self.back < SAFE_DISTANCE_LP or (self.save_y_pos - self.y) < 0.1:
                        if self.y < SAFE_LEFT_WALL:
                            self._cf.commander.send_position_setpoint(self.x, self.y - 0.1, DEFAULT_HEIGHT,
                                                                      self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                            time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                        else:
                            self.direction = "LEFT"
                            self._cf.commander.send_position_setpoint(self.x, self.y, DEFAULT_HEIGHT, self.y)
                            self.pass_obstacle = False
                            self.x_goback = False
                            break
                    else:
                        self._cf.commander.send_position_setpoint(self.x, self.y - 0.1, DEFAULT_HEIGHT,
                                                                  self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                        time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                        self.x_goback = True
                else:
                    self._cf.commander.send_position_setpoint(self.x - (self.x - self.save_x_pos), self.y - 0.1,
                                                              DEFAULT_HEIGHT,
                                                              self.y)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.5)  # Adjust sleep time based on responsiveness needs
                    self.pass_obstacle = False
                    self.x_goback = False
                    break

    def obstacle_avoidance_lp(self):
        """Simple obstacle avoidance routine."""
        if self.left < SAFE_DISTANCE_LP:
            if self.y > SAFE_LEFT_WALL:
                return (0, 0), (self.x, self.y)
            else:
                # If obstacle detected on the left, move right
                self.obstacle_avoidance_left()
        elif self.right < SAFE_DISTANCE_LP:
            if self.y < SAFE_RIGHT_WALL:
                return (0, 0), (self.x, self.y)
            else:
                # If obstacle detected on the left, move right
                self.obstacle_avoidance_right()

    def search_landing_pad(self):
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

        yaw_goal = np.sin(self.t*np.pi/20)
        yaw_goal = 20*yaw_goal
        yaw_goal = self.clip_angle(yaw_goal)
        dist = self.distance_2d(self.x,self.y,self.path_looking[self.path_looking_index,0],self.path_looking[self.path_looking_index,1])
        print(self.path)
        if dist < 0.1:
            self.path_looking_index +=1
        
        

        if self.path is None or self.t%20==0:
            
            start_node = [int((self.x +INITIAL_X )/res_pos +0.5) ,int((self.y + INITIAL_Y)/res_pos + 0.5)]
            end_node = [int((self.path_looking[self.path_looking_index,0] +INITIAL_X)/res_pos + 0.5),
                        int((self.path_looking[self.path_looking_index,1] +INITIAL_Y)/res_pos + 0.5)]

            print(start_node,end_node)
            path_test = a_star_search(np.array(self.occ_map), start_node, end_node,self)
            if path_test is not None:
                self.path = np.array(path_test) * res_pos
                self.path -= np.array([INITIAL_X, INITIAL_Y])
            
                self.path_iter_first = 0

        if self.path is not None:
            dist = self.distance_2d(self.x,self.y,self.path[self.path_iter_first][0],self.path[self.path_iter_first][1])
            if (dist<0.06) and self.path_iter_first <= len(self.path)-2:
                self.path_iter_first +=1
            self._cf.commander.send_position_setpoint(self.path[self.path_iter_first][0] , self.path[self.path_iter_first][1], DEFAULT_HEIGHT, yaw_goal)
            time.sleep(0.05)
        else:
            self._cf.commander.send_position_setpoint(self.x, self.y, DEFAULT_HEIGHT, yaw_goal)
            time.sleep(0.05)

            
        

        """ # Check for obstacle proximity
        if self.is_obstacle_close_lp():
            self.obstacle_avoidance_lp()
            time.sleep(0.05)
        else:
            if self.direction == "LEFT":
                if self.y < SAFE_LEFT_WALL + 0.1:
                    self._cf.commander.send_position_setpoint(self.x, self.y + 0.1, DEFAULT_HEIGHT,
                                                              yaw_goal)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                else:
                    self._cf.commander.send_position_setpoint(self.x + 0.2, self.y, DEFAULT_HEIGHT, yaw_goal)
                    self.direction = "RIGHT"
                    time.sleep(0.5)
            else:
                if self.y > SAFE_RIGHT_WALL - 0.1:
                    self._cf.commander.send_position_setpoint(self.x, self.y - 0.1, DEFAULT_HEIGHT,
                                                              yaw_goal)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
                else:
                    self._cf.commander.send_position_setpoint(self.x + 0.2, self.y, DEFAULT_HEIGHT,
                                                              yaw_goal)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    self.direction = "LEFT"
                    time.sleep(0.5)  # Adjust sleep time based on responsiveness needs
 """
    def update_path_looking(self):
        
        for i in range(len(self.path_looking)):

            path_point = self.path_looking[i] + np.array([INITIAL_X, INITIAL_Y])
            check = True
            #distance to move
            translate_dist = 0.1
            #define right or left side
            direction = True
            if path_point[1] < 1.5:
                direction = False
            loop_number = 0

            #while the number is not true
            while check and loop_number <= 10:
                #check if we are close to an obstacles
                path_point = self.path_looking[i] + np.array([INITIAL_X, INITIAL_Y])
                check = self.is_closed_obstacle(path_point)
                
                #if we are close to an obstacle
                if direction and check:
                    #new coordinate
                    offset = path_point[1] - translate_dist
                    #dont get outside the map
                    if offset < 0.1: offset = 0.1
                    #update the path
                    self.path_looking[i] = [path_point[0], offset] - np.array([INITIAL_X, INITIAL_Y])
                    
                elif check:
                    #new coordinate
                    offset = path_point[1] + translate_dist
                    #dont get outside the map
                    if offset > 2.9: offset = 2.9
                    #update the path
                    self.path_looking[i] = [path_point[0], offset] - np.array([INITIAL_X, INITIAL_Y])
                    
                loop_number += 1
        return

#check if we are too close to an obstacle     
    def is_closed_obstacle(self,path_point):
        #update the distance to check so that we dont go away from the side
        if path_point[0] > 4.7 - INITIAL_X:
            dist_x_top = 0
        else:
            dist_x_top = 2

        if path_point[1] <= 0.2 - INITIAL_Y:
            dist_y_right = 0
        else: 
            dist_y_right = 3

        if path_point[1] >2.7 - INITIAL_Y:
            dist_y_left = 0
        else:
            dist_y_left = 3
        dist_x_bottom = 2
        
        #convert to map reference frame
        
        x_on_map = int((path_point[0]+INITIAL_X )/res_pos +0.5)
        y_on_map = int((path_point[1]+INITIAL_Y )/res_pos +0.5)

        #check the surronding
        matches1 = np.where(self.occ_map[x_on_map - dist_x_bottom : x_on_map + dist_x_top, 
                                y_on_map - dist_y_right: y_on_map + dist_y_left] < 0.9)
        if len(matches1[0]) >0 :
            #no clean surrounding
            return True
        else: 
            #clean surrondings
            return False

    
    def print_state_info(self):
        state_repr: dict[FSM, str] = {
            FSM.INIT: f'',
            FSM.TAKE_OFF: f'down range: {int(self.down)} - z: {self.z:.3f} - az: {self.az:.3f}',
            FSM.ROTATE: f'',
            FSM.CROSS: f'x: {self.x:.3f}, y: {self.y:.3f}, yaw: {self.yaw:.3f}, FRONT: {self.front:.3f}',
            FSM.SEARCH: f'x: {self.x:.3f}, y: {self.y:.3f}, yaw: {self.yaw:.3f}, {self.az:.3f}',
            FSM.FOUND: f'down range: {int(self.down)} - z: {self.z:.3f} - az: {self.az:.3f}',
            FSM.CENTERING: f'down range: {int(self.down)} - z: {self.z:.3f} - az: {self.az:.3f}',
            FSM.LANDING: f'z: {self.z}, down range: {self.down}',
            FSM.STOP: f'',
            FSM.GOING_BACK: f'down range: {int(self.down)} - z: {self.z:.3f} - az: {self.az:.3f}',
            FSM.SPIRALING: f'x: {self.spiral_iter} {self.x}, y: {self.y}'
        }
        print(f'[{self.t}][{self.fsm}]: {state_repr[self.fsm]}')

    def log_data(self):
        self.logs.append(
            [self.t, self.fsm, self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz, self.front, self.back, self.left,
             self.right, self.up, self.down, self.az, self.default_direction, self.orientation, self.direction])

    def occupancy_map(self):
        measurements = np.array([self.front, self.left, self.back, self.right]) / 1000  # meters
        for j in range(4):  # 4 sensors
            yaw_sensor = np.deg2rad(self.yaw) + j * np.pi / 2  # yaw positive is counterclockwise
            measurement = measurements[j]
            for i in range(int(range_max / res_pos)):  # range is 2 meters
                dist = i * res_pos
                idx_x = int(np.round((self.x + INITIAL_X - min_x + dist * np.cos(yaw_sensor)) / res_pos, 0))
                idx_y = int(np.round((self.y + INITIAL_Y - min_y + dist * np.sin(yaw_sensor)) / res_pos, 0))

                # make sure the current set point is within the self.occ_map
                if idx_x < 0 or idx_x >= self.occ_map.shape[0] or idx_y < 0 or idx_y >= self.occ_map.shape[1] or dist > range_max:
                    break
        
                # update the self.occ_map
                if dist < measurement:
                    self.occ_map[idx_x, idx_y] += conf
                else:
                    self.occ_map[idx_x, idx_y] -= conf
                    break
        self.occ_map = np.clip(self.occ_map, -1, 1)  # certainty can never be more than 100%
         # Find border indices
        
        rows, cols = self.occ_map.shape
        border_indices_row = np.where((np.arange(rows) == 0) | (np.arange(rows) == rows - 1))
        border_indices_col = np.where((np.arange(cols) == 0) | (np.arange(cols) == cols - 1))

        # Set border elements to -1
        self.occ_map[border_indices_row[0], :] = -1  # Set first and last rows to -1
        self.occ_map[:, border_indices_col[0]] = -1  # Set first and last columns to -1


if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()
    le = LoggingExample(uri)
    cf = le._cf

    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

    # The Crazyflie lib doesn't contain anything to keep the application alive,
    # so this is where your application should do something. In our case we
    # are just waiting until we are disconnected.

    enable_log_data = True
    while le.is_connected:
        # time.sleep(0.01)
        t0 = time.perf_counter()
        le.t += 1
        le.fsm_update()
        le.actions[le.fsm]()
        t1 = time.perf_counter()
        print(f'Time: {t1 - t0:.3f}', end=' ')
        le.print_state_info()

        if enable_log_data:
            le.log_data()
        if le.t %20 == 0:
            plt.imshow(np.flip(le.occ_map, 1), vmin=-1, vmax=1, cmap='gray',
                    origin='lower')  # flip the map to match the coordinate system
            plt.scatter((max_y - (le.y +INITIAL_Y))/res_pos,(le.x + INITIAL_X)/res_pos , color='red')
            if le.path is not None and (np.any(le.path)):
            
                x_values = [(max_y - (point[1]+INITIAL_Y))/res_pos for point in le.path]
                y_values = [(point[0]+ INITIAL_X)/res_pos for point in le.path]
                plt.scatter(x_values, y_values, color='blue')
            
            
            x_values = [(max_y - (point[1]+INITIAL_Y))/res_pos for point in le.path_looking]
            y_values = [(point[0]+ INITIAL_X)/res_pos for point in le.path_looking]
            plt.scatter(x_values, y_values, color='yellow')

            plt.savefig("occ_map.png")
            plt.close()

    # Save the data
    if enable_log_data:
        df = pd.DataFrame(
            le.logs,
            columns=['t', 'fsm', 'x', 'y', 'z', 'yaw', 'vx', 'vy', 'vz',
                     'front', 'back', 'left', 'right', 'up', 'down', 'az',
                     'default_direction', 'orientation', 'direction'
                     ]
        )
        df.to_csv('logged_data.csv', index=False)

        # Save occupancy map
        plt.imshow(np.flip(le.occ_map, 1), vmin=-1, vmax=1, cmap='gray',
                   origin='lower')  # flip the map to match the coordinate system
        plt.savefig("occ_map.png")
        plt.close()
        print("Logging Successful")