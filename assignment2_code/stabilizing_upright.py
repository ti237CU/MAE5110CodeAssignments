import matplotlib.pyplot as plt
import numpy as np

def feedback_linearization(state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    theta = state[0]
    angular_vel = state[1]
    Kd = 2.0
    Kp = 1.0
    
    torque_min = -0.1 * mass * gravity * length
    torque_max = 0.05 * mass * gravity * length
    
    gravity_term = -(gravity/length) * np.sin(theta) * mass * length ** 2
    stabilizing_term = mass * length ** 2 * (- Kd * angular_vel - Kp * theta)
    control_input = gravity_term + stabilizing_term
    return np.clip(control_input, torque_min, torque_max)

def is_upright(state, tolerance):
    theta = state[0]
    angular_vel = state[1]
    
    return abs(theta) < tolerance and abs(angular_vel) < tolerance
    