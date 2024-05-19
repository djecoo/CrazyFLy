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
SAFE_DISTANCE = 400 #sensor reading
DEFAULT_HEIGHT = 0.3 #m
DEFAULT_VELOCITY = 0.1 #m/s
SAFE_RIGHT_WALL = 0.3 # meters
SAFE_LEFT_WALL = 2.7 # meters
INITIAL_X = 0.5
INITIAL_Y = 2.5
GOAL_X = 3.5

LANDING_PAD_SIZE = 0.3 # meters



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
        self.front = 0
        self.back = 0
        self.left = 0
        self.right = 0
        self.up = 0
        self.down = 0

        # Data for FSM
        self.search_direction = 0 # direction of search in radians
        self.found_pos = None # [x, y, z, yaw]
        self.centering_done = False
        self.centering_obervations = [] # [[x, y, down], ...]
        self.landing_pos = None # [x, y, z, yaw]
        self.square_positions = None # [[x, y], ...]

        self.t # Iteration time

        self.actions: dict[FSM, callable] = {
            FSM.INIT: lambda x: time.sleep(1),
            FSM.TAKE_OFF: self.take_off,
            FSM.ROTATE: self.rotate,
            FSM.SEARCH: self.search,
            FSM.FOUND: self.finding,
            FSM.CENTERING: self.centering,
            FSM.LANDING: self.landing,
            FSM.STOP: self.stop,
        }


        self._cf = Crazyflie(rw_cache='./cache')

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

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
        self._lg_stab.add_variable('range.down')
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
        """Callback from a the log API when data arrives"""
        print(f'[{timestamp}][{logconf.name}]: ', end='')
        for name, value in data.items():
            print(f'{name}: {value:3.3f} ', end='')
        print()

        self.x = data['stateEstimate.x']
        self.y = data['stateEstimate.y']
        self.z = data['stateEstimate.z']
        self.yaw = data['stabilizer.yaw']
        self.front = data['range.front']
        self.back = data['range.back']
        self.left = data['range.left']
        self.right = data['range.right']
        self.up = data['range.up']
        self.down = data['range.down']

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
        if self.fsm == FSM.INIT:
            self.fsm = FSM.TAKE_OFF
        elif self.fsm == FSM.TAKE_OFF:
            if self.z > 0.30:
                self.fsm = FSM.SEARCH
        elif self.fsm == FSM.SEARCH:
            if self.down < 100:
                self.fsm = FSM.FOUND
                self.found_pos = [self.x, self.y, self.z, self.yaw]
        elif self.fsm == FSM.FOUND:
            x, y, _, _ = self.found_pos
            square_positions = np.array([
                [0, 0],
                [LANDING_PAD_SIZE / 2, LANDING_PAD_SIZE / 2],
                [0, LANDING_PAD_SIZE, y],
                [LANDING_PAD_SIZE / 2, - LANDING_PAD_SIZE / 2],
                [0, 0]
            ])

            yaw = self.search_direction
            rotation_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])

            square_positions = np.dot(square_positions, rotation_matrix)
            self.square_positions = square_positions + np.array([x, y])

            self.fsm == FSM.CENTERING
        elif self.fsm == FSM.CENTERING:
            self.centering_obervations.append([self.x, self.y, self.down])
            if self.centering_done:
                # Calculate landing position
                observations = np.array(self.centering_obervations)
                f_observations = observations[observations[:, 2] < 100]
                land_pos = np.mean(f_observations[:, :2], axis=0)
                self.landing_pos = [land_pos[0], land_pos[1], self.found_pos[2], self.found_pos[3]]
                
                # Reinit centering variables
                self.centering_obervations = []
                self.centering_done = False

                self.fsm = FSM.LANDING
        elif self.fsm == FSM.LANDING:
            if self.down < 5:
                self.fsm = FSM.STOP

    def take_off(self):
        cf.commander.send_hover_setpoint(0, 0, 0, self.z + 0.1)
        time.sleep(0.1)

    def rotate(self):
        cf.commander.send_hover_setpoint(0, 0, 0.2, DEFAULT_HEIGHT)
        time.sleep(0.1)

    def search(self):
        cf.set_position_setpoint(self. x + DEFAULT_VELOCITY, self.z, 0, DEFAULT_HEIGHT)
        self.search_direction = np.arctan2(0, DEFAULT_VELOCITY)
        time.sleep(0.1)

    def finding(self):
        cf.set_position_setpoint(self.x, self.y, self.z, self.yaw)
        time.sleep(0.5)

    def centering(self):
        xf, yf, zf, yawf = self.found_pos
        if len(self.centering_obervations):
            x, y = self.square_positions[0]
        else:
            x, y = (xf, yf)
        cf.set_position_setpoint(x, y, zf, yawf)
        time.sleep(1)
        self.square_positions = self.square_positions[1:]

    def landing(self):
        x, y, _, yaw = self.landing_pos
        cf.set_position_setpoint(x, y, self.z - DEFAULT_VELOCITY, yaw)
        time.sleep(0.1)

    def stop(self):
        cf.commander.send_stop_setpoint()
        self._cf.close_link()
        time.sleep(0.1)
        exit(0)



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

        





