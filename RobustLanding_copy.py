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
from enum import Enum

import cflib.crtp  # noqa
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

uri = uri_helper.uri_from_env(default='radio://0/60/2M/E7E7E7E716')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

# Constants
SAFE_DISTANCE = 400  #sensor reading
DEFAULT_HEIGHT = 0.3  #m
DEFAULT_VELOCITY = 0.1  #m/s
SAFE_RIGHT_WALL = 0.3  # meters
SAFE_LEFT_WALL = 2.7  # meters
INITIAL_X = 0.0
INITIAL_Y = 0.0
GOAL_X = 3.5

LANDING_PAD_SIZE = 0.28  # meters


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


class LoggingExample:
    """
    Simple logging example class that logs the Stabilizer from a supplied
    link uri and disconnects after 5s.
    """

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """
        # Sensor values
        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0
        self.vx = 0.01
        self.vy = 0
        self.front = 0
        self.back = 0
        self.left = 0
        self.right = 0
        self.up = 0
        self.down = 0

        self.fsm = FSM.INIT

        # Data for FSM
        self.found_pos = None  # [x, y, z, yaw]
        self.square_positions = None  # [[x, y], ...]
        self.prev_square_pos = None
        self.centering_done = False
        self.centering_observations = []  # [[x, y, down], ...]
        self.landing_pos = None  # [x, y, z, yaw]

        self.landing_pad_reached = False

        self.t = 0  # Iteration time

        self.actions: dict[FSM, callable] = {
            FSM.INIT: lambda x: time.sleep(1),
            FSM.TAKE_OFF: self.take_off,
            FSM.ROTATE: self.rotate,  # Not used in this code
            FSM.SEARCH: self.search,
            FSM.FOUND: self.finding,
            FSM.CENTERING: self.centering,
            FSM.LANDING: self.landing,
            FSM.STOP: self.stop,
            FSM.GOING_BACK: self.going_back,
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
        self._lg_stab.add_variable('range.up')
        # self._lg_stab.add_variable('range.zrange')

        self._lg_stab2 = LogConfig(name='Velocities', period_in_ms=50)
        self._lg_stab2.add_variable('stateEstimate.vx', 'float')
        self._lg_stab2.add_variable('stateEstimate.vy', 'float')
        self._lg_stab2.add_variable('stateEstimate.vz', 'float')
        self._lg_stab2.add_variable('range.zrange', 'float')
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
        if self.t % 1 == 0:
            print(f'[{timestamp}][{logconf.name}]: ', end='')
            for name, value in data.items():
                print(f'{name}: {value:3.3f} ', end='')
            print()

        self.vx = data['stateEstimate.vx']
        self.vy = data['stateEstimate.vy']
        self.down = data['range.zrange']

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        if self.t % 1 == 0:  # Print every 30th iteration
            print(f'[{timestamp}][{logconf.name}]: ', end='')
            for name, value in data.items():
                print(f'{name}: {value:3.3f} ', end='')
            print()

        # Update sensor values
        self.x = data['stateEstimate.x']
        self.y = data['stateEstimate.y']
        self.z = data['stateEstimate.z']
        self.yaw = data['stabilizer.yaw']
        self.front = data['range.front']
        self.back = data['range.back']
        self.left = data['range.left']
        self.right = data['range.right']
        self.up = data['range.up']
        self.down = 1000 * data['stateEstimate.z']

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

    def _fsm_update(self):
        if self.up < 100 and self.fsm:
            self.landing_pos = [self.x, self.y, self.z, self.yaw]
            self.fsm = FSM.LANDING
            return

        if self.fsm == FSM.INIT:
            self.fsm = FSM.TAKE_OFF

        elif self.fsm == FSM.TAKE_OFF:
            if self.z > 0.30:
                self.fsm = FSM.GOING_BACK if self.landing_pad_reached else FSM.SEARCH

        elif self.fsm == FSM.SEARCH:
            if self.down < 280:
                self.fsm = FSM.FOUND
                self.found_pos = [self.x, self.y, self.z, self.yaw]
                self.prev_square_pos = self.found_pos[:2]

        elif self.fsm == FSM.FOUND:
            self.square_positions = self.compute_square_positions()
            self.fsm = FSM.CENTERING

        elif self.fsm == FSM.CENTERING:
            if self.centering_done:
                self.landing_pos = self.compute_landing_pos()
                self.centering_observations = []  # Reset centering observations for next landing
                self.centering_done = False

                self.fsm = FSM.LANDING

        elif self.fsm == FSM.LANDING:
            if self.down < 30:
                self.fsm = FSM.STOP if self.landing_pad_reached else FSM.TAKE_OFF
                self.landing_pad_reached = True


        elif self.fsm == FSM.GOING_BACK:
            if np.linalg.norm([self.x - INITIAL_X, self.y - INITIAL_Y]) < 0.3 and self.down < 280:
                self.found_pos = [self.x, self.y, self.z, self.yaw]
                self.fsm = FSM.FOUND

    def take_off(self):
        cf.commander.send_hover_setpoint(0, 0, 0, DEFAULT_HEIGHT)
        time.sleep(0.1)

    def rotate(self):
        cf.commander.send_hover_setpoint(0, 0, DEFAULT_HEIGHT, 0.2)
        time.sleep(0.1)

    def search(self):
        cf.commander.send_position_setpoint(self.x + DEFAULT_VELOCITY, self.y, DEFAULT_HEIGHT, 0)
        time.sleep(0.1)

    def going_back(self):
        vector = np.array([INITIAL_X - self.x, INITIAL_Y - self.y])
        vector = (vector / np.linalg.norm(vector)) * DEFAULT_VELOCITY
        cf.commander.send_position_setpoint(self.x + vector[0], self.y + vector[1], DEFAULT_HEIGHT, 0)
        time.sleep(0.1)

    def finding(self):
        cf.commander.send_position_setpoint(self.x, self.y, self.z, self.yaw)
        time.sleep(0.5)

    def centering(self):
        xf, yf, zf, yawf = self.found_pos
        x_prev, y_prev = self.prev_square_pos
        iter_num = 25  # Number of iterations to reach the next centering position
        for i in range(iter_num):
            if len(self.square_positions) > 0:
                x, y = self.square_positions[0]
                delta_x = x - x_prev
                delta_y = y - y_prev
                new_x = x_prev + delta_x * (i / iter_num)
                new_y = y_prev + delta_y * (i / iter_num)
            else:
                new_x, new_y = (xf, yf)
                self.centering_done = True
            self.centering_observations.append([self.x, self.y, self.down])
            print(f'Centering: {self.x:.3f}, {self.y:.3f}, {int(self.down)}')
            cf.commander.send_position_setpoint(new_x, new_y, zf, yawf)
            time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
        # print("Next centering position")
        if len(self.square_positions) > 0:  # Go to the next square position
            self.prev_square_pos = self.square_positions[0]
            self.square_positions = self.square_positions[1:]

    def landing(self):
        x, y, _, yaw = self.landing_pos
        cf.commander.send_position_setpoint(x, y, self.z - 0.05, yaw)
        time.sleep(0.1)

    def stop(self):
        cf.commander.send_stop_setpoint()
        self._cf.close_link()
        time.sleep(0.1)
        exit(0)

    def compute_landing_pos(self):
        observations = np.array(self.centering_observations)
        obs_delta = observations[2:, 2] - observations[:-2, 2]
        rising_edge = np.where(obs_delta > 35)[0]  # Going up on the pad
        falling_edge = np.where(obs_delta < -35)[0]  # Falling off the pad
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
        square_positions = np.array([
            [0, 0],
            [LANDING_PAD_SIZE / 2, - LANDING_PAD_SIZE / 2],
            [LANDING_PAD_SIZE, 0],
            [LANDING_PAD_SIZE / 2, LANDING_PAD_SIZE / 2],
            [0, 0]
        ])
        yaw = np.arctan2(self.vy, self.vx)  # TODO: Compute yaw by taking np.atan2(dy, dx) and check rotation matrix
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        square_positions = np.dot(square_positions, rotation_matrix)
        return square_positions + np.array([x, y])


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

    while le.is_connected:
        # time.sleep(0.01)
        le.t += 1
        le._fsm_update()
        le.actions[le.fsm]()

        print(f'[{le.t}][{le.fsm}]')
