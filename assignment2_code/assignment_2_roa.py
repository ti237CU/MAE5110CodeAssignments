import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from integrators import rk4 as integrator
from models import inverted_pendulum_walker as model
from assignment2_code import stabilizing_upright as controller

# Fixed controls for this visualization example.
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
}


timestep = 1e-3
sim_time = 20.0

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

theta_grid = np.linspace(-0.25, 0.25, 21)
angular_vel_grid = np.linspace(-0.7, 0.7, 21)
tol = 1e-3

theta_cur, angular_vel_cur = np.meshgrid(theta_grid, angular_vel_grid)
roa = np.zeros_like(theta_cur, dtype=bool)

for i in range(len(angular_vel_grid)):
    for j in range (len(theta_grid)):
        state = np.array([theta_grid[j], angular_vel_grid[i]])
        
        for step, t in enumerate(time_traj[:-1]):
            params["ankle_torque"] = controller.feedback_linearization(state, params)
            next_state = integrator.integrate(
                t,
                state,
                timestep,
                model.dynamics,
                params
            )
            if controller.is_upright(next_state, tol):
                roa[i, j] = True
                break
            
            state = next_state
            
plt.figure(figsize=(8,6))

plt.contourf(
    theta_cur,
    angular_vel_cur,
    roa.astype(int),
    levels = [-0.5, 0.5, 1.5],
    colors=["blue","green"]
)

plt.xlabel(r"$\theta$ [rad]")
plt.ylabel(r"$\dot{\theta}$ [rad/s]")
plt.title("Region of Attraction (RoA)")

legend_elements = {
    Patch(facecolor="green", label = "converged"),
    Patch(facecolor="blue", label = "not converged / timed out")
}
plt.legend(handles=legend_elements)

plt.show()


