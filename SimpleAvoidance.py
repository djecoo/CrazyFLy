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
DEFAULT_HEIGHT = 0.3 #m
DEFAULT_VELOCITY = 0.1 #m/s

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
        self.initial_position_reached = False
        self.x_target = 3.
        

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
            if self.left < self.right:
                # Move left
                return (0, DEFAULT_VELOCITY), (self.x, self.y + DEFAULT_VELOCITY)
            else:
                # Move right
                return (0, -DEFAULT_VELOCITY), (self.x, self.y - DEFAULT_VELOCITY)
        elif self.left < SAFE_DISTANCE:
            # If obstacle detected on the left, move right
            return (0, -DEFAULT_VELOCITY), (self.x, self.y - DEFAULT_VELOCITY)
        elif self.right < SAFE_DISTANCE:
            # If obstacle detected on the right, move left
            return (0, DEFAULT_VELOCITY), (self.x, self.y + DEFAULT_VELOCITY)
        elif self.back < SAFE_DISTANCE:
            # If obstacle detected at the back, move forward
            return (DEFAULT_VELOCITY, 0), (self.x + DEFAULT_VELOCITY, self.y)
        else:
            # No obstacles detected, maintain current velocity and position
            return (0, 0), (self.x,self.y)
    def move_command(self):
        # Start by taking off and hovering at the initial position.
        

        if not self.initial_position_reached:
            
            if self.x >= self.x_target:
                self.initial_position_reached = True
                print("Reached 3 meters in X")
                self.land = True
                return
            else:
                # Check for obstacle proximity
                if self.is_obstacle_close():
                    velocity, position = self.obstacle_avoidance_routine()
                    self._cf.commander.send_position_setpoint(position[0], position[1], DEFAULT_HEIGHT,0)
                    time.sleep(0.05)
                else:
                    self._cf.commander.send_position_setpoint(self.x + 0.1, self.y, DEFAULT_HEIGHT, 0)  # Target x=3 meters, y=0 (no change), z=0.5 meters
                    time.sleep(0.05)  # Adjust sleep time based on responsiveness needs
    


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
    try: 
        while le.is_connected:
            time.sleep(0.01)
            if le.back < 300:
                le.land  = True
            if le.land:
                for _ in range(20):
                    print('landing')
                    cf.commander.send_hover_setpoint(0, 0, 0, 0.5)
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
                    cf.commander.send_hover_setpoint(0, 0, 0, 0.5)
                    time.sleep(0.1)
                le.lift_off = True
                print('lift off')
            else:
                le.move_command()
                le.iter +=1
                print(le.iter)
    except KeyboardInterrupt:
        cf.commander.send_stop_setpoint()

        




