## This is course material for Introduction to Modern Artificial Intelligence
## Class 20 Example code: PID_control.py
## Author: Allen Y. Yang,  Intelligent Racing Inc.
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import random
import numpy as np
import matplotlib.pyplot as plt

class Vehicle2D(object):
    def __init__(self, length=10.0):
        """
        Creates 2D vehicle model and initializes location (0,0) & orientation 0.
        """
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0.0
        self.length = length
        self.steering_noise = 0.0
        self.distance_noise = 0.0
        self.steering_drift = 0.0

    def set(self, x, y, orientation):
        """
        Sets the vehicle coordinates and orientation.
        """
        self.x = x
        self.y = y
        self.orientation = orientation % (2.0 * np.pi)

    def set_noise(self, steering_noise, distance_noise):
        """
        Sets the noise parameters.
        """
        # makes it possible to change the noise parameters
        self.steering_noise = steering_noise
        self.distance_noise = distance_noise

    def set_steering_drift(self, drift):
        """
        Sets the systematical steering drift parameter
        """
        self.steering_drift = drift

    def move(self, steering, distance, tolerance=0.001, max_steering_angle=np.pi / 4.0):
        """
        steering = front wheel steering angle, limited by max_steering_angle(max 90 degree)
        distance = total distance driven, most be non-negative
        """
        if max_steering_angle >np.pi/2:
            raise ValueError("Max steering angle cannot be greater than 90-degree.")

        if steering > max_steering_angle:
            steering = max_steering_angle
        if steering < -max_steering_angle:
            steering = -max_steering_angle
        if distance < 0.0:
            distance = 0.0

        # apply noise
        steering2 = random.gauss(steering, self.steering_noise)
        distance2 = random.gauss(distance, self.distance_noise)

        # apply steering drift
        steering2 += self.steering_drift

        # Execute motion
        turn = np.tan(steering2) * distance2 / self.length

        if abs(turn) < tolerance:
            # approximate by straight line motion
            self.x += distance2 * np.cos(self.orientation)
            self.y += distance2 * np.sin(self.orientation)
            self.orientation = (self.orientation + turn) % (2.0 * np.pi)
        else:
            # approximate bicycle model for motion
            radius = distance2 / turn
            cx = self.x - (np.sin(self.orientation) * radius)
            cy = self.y + (np.cos(self.orientation) * radius)
            self.orientation = (self.orientation + turn) % (2.0 * np.pi)
            self.x = cx + (np.sin(self.orientation) * radius)
            self.y = cy - (np.cos(self.orientation) * radius)

    def compute_error(self, track):
        min_distance = float('inf')
        for i in range(len(track)):
            distance = np.linalg.norm(np.array([self.x, self.y])- np.array(track[i]))
            if distance<min_distance:
                min_distance = distance
                error = self.y - track[i][1]  # This error only assumes cte based on y-axis. Further extension is possible

        return error

    def run_PID(self, track, K_p, K_d, K_i, n=500, speed=1.0):
        x_trajectory = []
        y_trajectory = []
    
        prev_error = self.compute_error(track)
        cumulative_error = 0
        for _ in range(n):
            current_error = self.compute_error(track)
            cumulative_error += current_error
            dev = current_error - prev_error
            prev_error = current_error
            steer = -K_p * current_error - K_d * dev - K_i*cumulative_error
            self.move(steer, speed)
            x_trajectory.append(self.x)
            y_trajectory.append(self.y)
        
        return x_trajectory, y_trajectory

    def __repr__(self):
        return '[x=%.5f y=%.5f orient=%.5f]' % (self.x, self.y, self.orientation)

vehicle = Vehicle2D()
vehicle.set(0, 1, 0)
vehicle.set_steering_drift(10/180.*np.pi)

track_length = 1000
targets = []; track_x = []; track_y = []
for i in range(300):
    targets.append([i, 0]); track_x.append(i); track_y.append(0)
for i in range(300,600):
    targets.append([i, 2]); track_x.append(i); track_y.append(2)
for i in range(600,track_length):
    targets.append([i, 0]); track_x.append(i); track_y.append(0)

K_p = 0.2; K_d = 3; K_i = 0.0005
x_trajectory, y_trajectory = vehicle.run_PID(targets, K_p, K_d, K_i, track_length)  
n = len(x_trajectory)

controller_legend = 'PID controller' if K_p!=0 and K_d!=0 and K_i!=0 else 'PD controller'\
     if K_p!=0 and K_d!=0 else 'P controller' if K_p!=0 else 'unknown controller'
fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
ax1.plot(x_trajectory, y_trajectory, 'g', linewidth=4, label= controller_legend)
ax1.plot(track_x, track_y, 'r', label='target')
plt.legend()
plt.show()