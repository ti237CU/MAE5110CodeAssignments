from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch
import simulate_step as simulate


base_dir = Path(__file__).resolve().parent


params = {
    "gravity": 9.81,
    "length": 1.0,
    "mass": 1.0,
    "incline": 0.06,
    "angle_of_attack": np.pi / 8,
    "ankle_torque": 0.0,
}

# Selected grid size to be used
resolution = 50


# Load roa data
roa_data = np.load(base_dir / "roa_data.npz")

theta_grid = roa_data["theta_grid"]
angular_vel_grid = roa_data["angular_vel_grid"]
roa = roa_data["roa"]


lookup = np.load(
    base_dir
    / "resolution_study"
    / f"lookup_{resolution}x"
      f"{resolution}.npz"
)

omega_grid = lookup["omega_grid"]
alpha_grid = lookup["alpha_grid"]

next_omega_table = lookup["next_omega_table"]
reaches_roa = lookup["reaches_roa"]
already_in_roa = lookup["already_in_roa"]
steps_to_roa = lookup["steps_to_roa"]
control_alpha = lookup["control_alpha"]


# Pick initial condition for at least 3 steps
three_step_candidates = np.where(steps_to_roa == 3)[0]

if three_step_candidates.size > 0:
    initial_i = three_step_candidates[len(three_step_candidates) // 2]

else:
    candidates = np.where(steps_to_roa >= 3)[0]

    if candidates.size == 0:
        raise RuntimeError("No state requires at least 3 steps.")

    initial_i = candidates[0]


initial_omega = omega_grid[initial_i]

print(
    f"Chosen initial omega = "
    f"{initial_omega:.4f} rad/s"
)

print(
    f"Minimum predicted steps = "
    f"{steps_to_roa[initial_i]}"
)


max_horizon = 20

n_omega = len(omega_grid)
n_alpha = len(alpha_grid)

exact_reachable = np.zeros((max_horizon + 1, n_omega), dtype=bool)
exact_action = np.full((max_horizon + 1, n_omega), -1,dtype=int)
exact_reachable[0, :] = already_in_roa


for h in range(1, max_horizon + 1):
    for i in range(n_omega):
        if already_in_roa[i]:
            continue

        for j in range(n_alpha):

            if h == 1:
                if reaches_roa[i, j]:
                    exact_reachable[h, i] = True
                    exact_action[h, i] = j
                    break

            else:
                next_omega = (next_omega_table[i, j])

                if np.isnan(next_omega):
                    continue

                if (next_omega < omega_grid[0] or next_omega > omega_grid[-1]):
                    continue

                next_i = np.argmin(np.abs(omega_grid - next_omega))

                if exact_reachable[h - 1, next_i]:
                    exact_reachable[h, i] = True
                    exact_action[h, i] = j
                    break


max_steps_to_roa = np.full(n_omega,-1,dtype=int)

for i in range(n_omega):
    possible_steps = np.where(exact_reachable[:, i])[0]

    if possible_steps.size > 0:
        max_steps_to_roa[i] = (possible_steps.max())

initial_max_steps = (max_steps_to_roa[initial_i])

print(f"Maximum steps found for initial omega = {initial_max_steps}")


if initial_max_steps == max_horizon:
    print("WARNING: maximum reached the search horizon. ")

def append_segment(
    all_time,
    all_states,
    segment_time,
    segment_states
):

    if len(all_time) == 0:
        all_time.extend(segment_time.tolist())
        all_states.extend(segment_states.tolist())

    else:
        offset = all_time[-1]
        all_time.extend((offset+ segment_time[1:]).tolist())
        all_states.extend(segment_states[1:].tolist())



def rollout_minimum(initial_omega):

    current_omega = initial_omega

    all_time = []
    all_states = []

    poincare_omegas = [current_omega]
    alpha_history = []


    for step in range(max_horizon):
        current_i = np.argmin(np.abs(omega_grid - current_omega))

        if steps_to_roa[current_i] == 0:
            break

        alpha = control_alpha[current_i]

        if np.isnan(alpha):
            break

        alpha_history.append(alpha)

        (status, result, segment_time, segment_states) = simulate.simulate_step_history(current_omega,alpha,theta_grid,angular_vel_grid,roa,params)

        append_segment(all_time, all_states, segment_time, segment_states)


        if status in ("roa", "already in roa"):

            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                step + 1
            )


        if status == "poincare":

            current_omega = result
            poincare_omegas.append(current_omega)

        else:
            break


    return (
        np.asarray(all_time),
        np.asarray(all_states),
        np.asarray(poincare_omegas),
        np.asarray(alpha_history),
        -1
    )


def rollout_exact_steps(initial_omega,number_of_steps):

    current_omega = initial_omega
    remaining = number_of_steps

    all_time = []
    all_states = []

    poincare_omegas = [current_omega]
    alpha_history = []


    while remaining > 0:

        current_i = np.argmin(np.abs(omega_grid - current_omega))
        j = exact_action[remaining,current_i]

        if j < 0:
            print("No exact-step action found.")
            break

        alpha = alpha_grid[j]
        alpha_history.append(alpha)

        (status, result, segment_time, segment_states) = simulate.simulate_step_history(current_omega, alpha,theta_grid,angular_vel_grid,roa,params)

        append_segment(all_time, all_states, segment_time, segment_states)

        if status in ("roa","already in roa"):
            if remaining != 1:
                print("WARNING: actual rollout entered the RoA earlier than the lookup table predicted.")

            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                number_of_steps
            )


        if status == "poincare":
            current_omega = result
            poincare_omegas.append(current_omega)
            remaining -= 1

        else:
            print("Maximum-step rollout failed.")
            break


    return (
        np.asarray(all_time),
        np.asarray(all_states),
        np.asarray(poincare_omegas),
        np.asarray(alpha_history),
        -1
    )



(min_time, min_states, min_poincare, min_alpha, min_actual_steps) = rollout_minimum(initial_omega)
(max_time, max_states, max_poincare, max_alpha, max_actual_steps) = rollout_exact_steps(initial_omega, initial_max_steps)

print(f"\nMinimum-step rollout = {min_actual_steps} steps")
print(f"Maximum-step rollout = {max_actual_steps} steps")


theta_mesh, omega_mesh = np.meshgrid(theta_grid, angular_vel_grid)

# Plot trajectory for an initial condition that requires at least 3 steps
plt.figure(figsize=(9, 6))
plt.contourf(theta_mesh, omega_mesh, roa.astype(int), levels=[0.5, 1.5],alpha=0.15)

if min_states.size > 0:
    plt.plot(min_states[:, 0], min_states[:, 1],label=(f"Lookup-policy trajectory {min_actual_steps} steps)"))

if max_states.size > 0:
    plt.plot(max_states[:, 0], max_states[:, 1], label=(f"Maximum-step trajectory ({max_actual_steps} steps)"))

plt.scatter(
    np.zeros(len(min_poincare)),
    min_poincare,
    marker="o",
    label="Lookup-policy Poincaré crossings"
)

plt.scatter(
    np.zeros(len(max_poincare)),
    max_poincare,
    marker="x",
    label="Maximum-policy Poincaré crossings"
)

roa_patch = Patch(
    alpha=0.15,
    label="Standing Region of Attraction"
)
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(roa_patch)

labels.append("Standing Region of Attraction")

plt.xlabel(r"$\theta$ [rad]")
plt.ylabel(r"$\dot{\theta}$ [rad/s]")

plt.title(f'Walking Trajectories from $\\dot{{\\theta}}_0={initial_omega:.3f}$ rad/s')
plt.legend(handles, labels)
plt.grid()

plt.show()


# Plot visualization fo how man steps it takes to get to standstill for a given initial condition
valid = steps_to_roa >= 0

plt.figure(figsize=(8, 5))

plt.step(omega_grid[valid], steps_to_roa[valid], where="mid")
plt.scatter(omega_grid[valid], steps_to_roa[valid])

plt.xlabel(r"Initial Poincaré Velocity $\dot{\theta}_0$ [rad/s]")
plt.ylabel("Predicted Walking Steps Before Entering RoA")
plt.title("Predicted Steps Required to Reach Standstill Region")

plt.yticks(np.arange(0, np.max(steps_to_roa[valid]) + 1))

plt.grid()
plt.tight_layout()
plt.show()