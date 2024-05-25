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

uri = uri_helper.uri_from_env(default='radio://0/60/2M/E7E7E7E716')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

# Constants
SAFE_DISTANCE = 400  #sensor reading
DEFAULT_HEIGHT = 0.3  #m
DEFAULT_VELOCITY = 0.1  #m/s
SAFE_RIGHT_WALL = 0.3  # meters
SAFE_LEFT_WALL = 2.7  # meters
INITIAL_X = 0.3
INITIAL_Y = 1.5
GOAL_X = 1

Z_REF_COUNTER = 20
LANDING_PAD_SIZE = 0.30  # meters$

range_max = 2  # meters
res_pos = 0.2  # meters
conf = 0.1
min_x, max_x = 0 - INITIAL_X, 5 - INITIAL_X
min_y, max_y = 0 - INITIAL_Y, 3 - INITIAL_Y



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
        self.z_ref_counter = Z_REF_COUNTER
        self.az = 0

        self.down_buffer_data = []

        self.fsm = FSM.INIT

        # Data for FSM
        self.found_pos = None  # [x, y, z, yaw]
        self.pattern_position = np.array([])  # [[x, y], ...]
        # self.prev_square_pos = None
        self.centering_observations = []  # [[x, y, down], ...]
        self.landing_pos = None  # [x, y, z, yaw]
        self.landing_pad_reached = False
        self.spiral_coord = normalize_spiral(archimedean_spiral(num_points=500), fixed_norm=0.02)

        self.t = 0  # Iteration time

        self.occ_map = np.zeros((int(5 / res_pos), int(3 / res_pos)))
        self.circle_points = normalize_spiral(circle(radius=LANDING_PAD_SIZE / 1.5), fixed_norm=0.01)

        self.actions: dict[FSM, callable] = {
            FSM.INIT: lambda x: time.sleep(1),
            FSM.TAKE_OFF: self.take_off,
            FSM.ROTATE: self.rotate,  # Not used in this code
            FSM.CROSS: self.cross,  # Not used in this code
            FSM.SEARCH: self.search,
            FSM.FOUND: self.finding,
            FSM.CENTERING: self.centering,
            FSM.LANDING: self.landing,
            FSM.STOP: self.stop,
            FSM.GOING_BACK: self.going_back,
            FSM.SPIRALING: self.spiraling2,
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
        if self.t % 1 == -1:
            print(f'[{timestamp}][{logconf.name}]: ', end='')
            for name, value in data.items():
                print(f'{name}: {value:3.3f} ', end='')
            print()

        self.vx = data['stateEstimate.vx']
        self.vy = data['stateEstimate.vy']
        self.vz = data['stateEstimate.vz']
        self.down = data['range.zrange']
        self.az = data['stateEstimate.az']
        self.down_buffer.append(self.down)

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        if self.t % 1 == -1:  # Print every 30th iteration
            print(f'[{timestamp}][{logconf.name}]: ', end='')
            for name, value in data.items():
                print(f'{name}: {value:3.3f} ', end='')
            print()

        # Update sensor values
        self.x = data['stateEstimate.x']
        self.y = data['stateEstimate.y']
        self.z = data['stateEstimate.z']
        self.yaw = data['stabilizer.yaw']
        self.back = min(data['range.back'], 2000)
        self.left = min(data['range.left'], 2000)
        self.front = min(data['range.front'], 2000)
        self.right = min(data['range.right'], 2000)
        self.up = data['range.up']

        self.occupancy_map()

        # global DEFAULT_HEIGHT
        # if self.z_ref_counter > 0:
        #     self.z_ref_counter -= 1
        # else:
        #     if self.down > 575:
        #         if self.z_ref == 0.2:
        #             self.z_ref_counter = Z_REF_COUNTER
        #             self.z_ref = 0.3
        #     else:
        #         if self.z_ref == 0.3:
        #             self.z_ref_counter = Z_REF_COUNTER
        #             self.z_ref = 0.2
        # self.down = 1000 * data['stateEstimate.z']

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
            if self.z > 0.20:
                self.fsm = FSM.GOING_BACK if self.landing_pad_reached else FSM.CROSS
                self.take_off_iter = 0

        elif self.fsm == FSM.CROSS:
            if self.x + INITIAL_X > GOAL_X:
                self.fsm = FSM.SEARCH

        elif self.fsm == FSM.SEARCH or self.fsm == FSM.SPIRALING or self.fsm == FSM.GOING_BACK:
            if self.az > 0.08:
                if self.fsm != FSM.GOING_BACK or self.x < 0.3:
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
                if np.linalg.norm([self.x, self.y]) < 0.1:
                    self.fsm = FSM.SPIRALING

        elif self.fsm == FSM.FOUND:
            # self.square_positions = self.compute_square_positions()
            # print(f'Square positions: {self.square_positions}')
            if self.finding_iter == 5:
                self.fsm = FSM.CENTERING
                self.finding_iter = 0

        elif self.fsm == FSM.CENTERING:
            if self.pattern_iter == len(self.pattern_position):
                # Save centering observations as dataframe
                down_buffer_df = pd.DataFrame(np.array(self.down_buffer_data))
                down_buffer_df.to_csv('down_buffer.csv', index=False)
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
        cf.commander.send_hover_setpoint(0, 0, 0, DEFAULT_HEIGHT)
        time.sleep(0.05)

    def rotate(self):
        cf.commander.send_hover_setpoint(0, 0, 30, DEFAULT_HEIGHT)
        time.sleep(0.05)

    def cross(self):
        cf.commander.send_position_setpoint(self.x + DEFAULT_VELOCITY, self.y, DEFAULT_HEIGHT, 0)
        time.sleep(0.05)

    def search(self):
        cf.commander.send_position_setpoint(self.x + DEFAULT_VELOCITY, self.y, DEFAULT_HEIGHT, 0)
        time.sleep(0.05)

    def going_back(self):
        vector = -np.array([self.x, self.y])
        if np.linalg.norm(vector) > DEFAULT_VELOCITY:
            vector = (vector / np.linalg.norm(vector)) * DEFAULT_VELOCITY
            cf.commander.send_position_setpoint(self.x + vector[0], self.y + vector[1], DEFAULT_HEIGHT, 0)
        else:
            cf.commander.send_position_setpoint(0, 0, DEFAULT_HEIGHT, 0)
        time.sleep(0.05)

    def spiraling(self):
        self.spiral_iter += 1
        max_vel = DEFAULT_VELOCITY * 2
        vel = min((self.spiral_iter / 200) * max_vel, max_vel)
        cf.commander.send_hover_setpoint(vel, 0, 50, DEFAULT_HEIGHT)
        time.sleep(0.05)

    def spiraling2(self):
        x, y = self.spiral_coord[self.spiral_iter]
        self.spiral_iter += 1
        if self.spiral_iter == len(self.spiral_coord):
            self.landing_pos = [x, y, DEFAULT_HEIGHT, 0]
            self.fsm = FSM.LANDING
            print("Spiral finished----------------------------")
        cf.commander.send_position_setpoint(x, y, DEFAULT_HEIGHT, 0)
        time.sleep(0.05)

    def finding(self):
        xf, xf, _, yaw = self.found_pos
        x, y = self.pattern_position[0]
        # for i in range(20):
        cf.commander.send_position_setpoint(x, y, DEFAULT_HEIGHT, yaw)
        self.finding_iter += 1
        time.sleep(0.05)

    def centering(self):
        next_pos = self.pattern_position[self.pattern_iter]
        x, y, _, yaw = self.found_pos

        self.centering_observations.append([self.x, self.y, self.down, self.az])
        self.down_buffer_data.append([d for d in self.down_buffer])
        # z_go = self.z * 0.99 + (DEFAULT_HEIGHT + 0.05) * 0.01
        cf.commander.send_position_setpoint(next_pos[0], next_pos[1], DEFAULT_HEIGHT, yaw)
        self.pattern_iter += 1
        time.sleep(0.05)

    # def centering2(self):
    #     self.center_iter += 1
    #     iter_num = 40  # Number of iterations to reach the next centering position
    #     xf, yf, _, yawf = self.found_pos
    #     x_prev, y_prev = self.prev_square_pos
    #     x, y = self.pattern_position[0]
    #     delta_x = x - x_prev
    #     delta_y = y - y_prev
    #     if len(self.pattern_position) > 0:
    #         new_x = x_prev + delta_x * (self.center_iter / iter_num)
    #         new_y = y_prev + delta_y * (self.center_iter / iter_num)
    #     else:
    #         new_x, new_y = (xf, yf)
    #     self.centering_observations.append([self.x, self.y, self.down])
    #     # print(f'Centering: {self.x:.3f}, {self.y:.3f}, {int(self.down)}, {int(self.down_range)}')
    #     # print(f'Down buffer diff: {[d - self.down for d in self.down_buffer]}')
    #     self.down_buffer_data.append([d for d in self.down_buffer])
    #     cf.commander.send_position_setpoint(new_x, new_y, DEFAULT_HEIGHT, yawf)
    #     time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
    #     # print("Next centering position")
    #     if len(self.pattern_position) > 0 and self.center_iter > iter_num:
    #         # Go to the next square position
    #         self.prev_square_pos = self.pattern_position[0]
    #         self.pattern_position = self.pattern_position[1:]
    #         self.center_iter = 0

    def landing(self):
        x, y, _, yaw = self.landing_pos

        for i in range(10):
            cf.commander.send_position_setpoint(x, y, DEFAULT_HEIGHT, yaw)
            time.sleep(0.05)

        iter_num = 30
        for i in range(iter_num + 20):
            cf.commander.send_position_setpoint(x, y, DEFAULT_HEIGHT - (DEFAULT_HEIGHT / iter_num) * i, yaw)
            if i > iter_num:
                cf.commander.send_stop_setpoint()
                # print('Landing')
                self.print_state_info()
            time.sleep(0.1)

    def stop(self):
        cf.commander.send_stop_setpoint()
        self._cf.close_link()
        time.sleep(0.05)


    def compute_landing_pos2(self):
        observations = np.array(self.centering_observations)
        azs = observations[:, -1]
        mask = np.full(len(observations), np.nan)
        mask[azs > 0.08] = 1
        mask[azs < 0.08] = -1

        window_size = 10
        for i in range(window_size, len(mask)):
            pass

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

    def compute_square_positions(self):
        x, y, _, _ = self.found_pos
        vx, vy = self.found_vel
        square_positions = np.array([
            [0, 0],
            [LANDING_PAD_SIZE / 2, - LANDING_PAD_SIZE / 2],
            [LANDING_PAD_SIZE / 2, LANDING_PAD_SIZE / 2],
            [LANDING_PAD_SIZE, 0],
            [0, 0]
        ])
        # square_positions = np.array([
        #     [0, 0],
        #     [LANDING_PAD_SIZE, 0],
        #     [LANDING_PAD_SIZE / 2, -LANDING_PAD_SIZE],
        #     [LANDING_PAD_SIZE / 2, LANDING_PAD_SIZE],
        #     [-LANDING_PAD_SIZE/2, 0],
        #     [LANDING_PAD_SIZE / 2, -LANDING_PAD_SIZE],
        #     [LANDING_PAD_SIZE/2, LANDING_PAD_SIZE/2],
        #     [0, 0]
        # ])
        yaw = np.arctan2(vy, vx)
        rotation_matrix = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        square_positions = np.dot(square_positions, rotation_matrix)
        return square_positions + np.array([x, y])

    def print_state_info(self):

        state_repr: dict[FSM, str] = {
            FSM.INIT: f'',
            FSM.TAKE_OFF: f'down range: {int(self.down)} - z: {self.z:.3f} - height: {DEFAULT_HEIGHT:.3f} - ref: {self.z_ref:.3f}, az: {self.az:.3f}',
            FSM.ROTATE: f'',
            FSM.CROSS: f'down range: {int(self.down)} - z: {self.z:.3f} - height: {DEFAULT_HEIGHT:.3f} - ref: {self.z_ref:.3f}, az: {self.az:.3f}',
            FSM.SEARCH: f'down range: {int(self.down)} - z: {self.z:.3f} - height: {DEFAULT_HEIGHT:.3f} - ref: {self.z_ref:.3f}, az: {self.az:.3f}',
            FSM.FOUND: f'down range: {int(self.down)} - z: {self.z:.3f} - height: {DEFAULT_HEIGHT:.3f} - ref: {self.z_ref:.3f}, az: {self.az:.3f}',
            FSM.CENTERING: f'down range: {int(self.down)} - z: {self.z:.3f} - height: {DEFAULT_HEIGHT:.3f} - ref: {self.z_ref:.3f}, az: {self.az:.3f}',
            FSM.LANDING: f'z: {self.z}, down range: {self.down}',
            FSM.STOP: f'',
            FSM.GOING_BACK: f'down range: {int(self.down)} - z: {self.z:.3f} - height: {DEFAULT_HEIGHT:.3f} - ref: {self.z_ref:.3f}, az: {self.az:.3f}',
            FSM.SPIRALING: f'x: {self.spiral_iter} {self.x}, y: {self.y}'
        }
        print(f'[{self.t}][{self.fsm}]: {state_repr[self.fsm]}')

    def log_data(self):
        self.logs.append(
            [self.t, self.fsm, self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz, self.front, self.back, self.left,
             self.right, self.up, self.down, self.down_buffer, self.z_ref, self.az])

    def occupancy_map(self):
        for j in range(4):  # 4 sensors
            yaw_sensor = self.yaw + j * np.pi / 2  # yaw positive is counter clockwise
            if j == 0:
                measurement = self.front / 1000
            elif j == 1:
                measurement = self.left / 1000
            elif j == 2:
                measurement = self.back / 1000
            elif j == 3:
                measurement = self.right / 1000

            for i in range(int(range_max / res_pos)):  # range is 2 meters
                dist = i * res_pos
                idx_x = int(np.round((self.x - min_x + dist * np.cos(yaw_sensor)) / res_pos, 0))
                idx_y = int(np.round((self.y - min_y + dist * np.sin(yaw_sensor)) / res_pos, 0))

                # make sure the current_setpoint is within the self.occ_map
                if idx_x < 0 or idx_x >= self.occ_map.shape[0] or idx_y < 0 or idx_y >= self.occ_map.shape[1] or dist > range_max:
                    break

                # update the self.occ_map
                if dist < measurement:
                    self.occ_map[idx_x, idx_y] += conf
                else:
                    self.occ_map[idx_x, idx_y] -= conf
                    break
        self.occ_map = np.clip(self.occ_map, -1, 1)  # certainty can never be more than 100%



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

    # Save the data
    if enable_log_data:
        df = pd.DataFrame(
            le.logs,
            columns=['t', 'fsm', 'x', 'y', 'z', 'yaw', 'vx', 'vy', 'vz',
                     'front', 'back', 'left', 'right', 'up', 'down',
                     'down_buffer', 'z_ref', 'az']
        )
        df.to_csv('logged_data.csv', index=False)

        # Save occupancy map
        plt.imshow(np.flip(le.occ_map, 1), vmin=-1, vmax=1, cmap='gray',
                   origin='lower')  # flip the map to match the coordinate system
        plt.savefig("occ_map.png")
        plt.close()
        print("Logging Succesfull")