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
    
    return ((np.abs(theta) < tolerance) & (np.abs(angular_vel) < tolerance))

def in_roa(state, theta_grid, angular_vel_grid, roa):
    theta = state[0]
    angular_vel = state[1]
    
    # Check if it is in roa
    if (
        theta_grid[0] <= theta <= theta_grid[-1]
        and
        angular_vel_grid[0] <= angular_vel <= angular_vel_grid[-1]
    ):
        j = np.argmin(np.abs(theta_grid - theta))
        i = np.argmin(np.abs(angular_vel_grid - angular_vel))

        return bool(roa[i, j])

    return False

def roa_event_guard(previous_state, next_state, theta_grid, angular_vel_grid, roa):
    previous_inside = in_roa(previous_state, 
                             theta_grid, 
                             angular_vel_grid, 
                             roa)
    
    next_inside = in_roa(next_state,
                         theta_grid,
                         angular_vel_grid,
                         roa)
    
    return (not previous_inside) and next_inside

def poincare_section(previous_state, next_state):
    previous_theta  = previous_state[0]
    next_theta = next_state[0]
    
    return(
        previous_theta < 0 and
        next_theta >= 0 and
        next_state[1] > 0
    )