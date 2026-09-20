import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import simulate_step as simulate

params = {
    "gravity": 9.81,
    "length": 1.0,
    "mass": 1.0,
    "incline": 0.06,
    "angle_of_attack": np.pi / 8,
    "ankle_torque": 0.0,
}

roa_data = np.load( "assignment_2/roa_data.npz")
theta_grid = roa_data["theta_grid"]
angular_vel_grid = roa_data["angular_vel_grid"]
roa = roa_data["roa"]

resolution_dir = Path("assignment_2/resolution_study")
resolutions = [10, 20, 30, 40, 50, 60, 70, 80,]

omega_min = 0.0
omega_max = np.sqrt(2 * params["gravity"] / params["length"])

alpha_min = np.pi / 8
alpha_max = np.pi / 7

n_validation_omega = 50
n_validation_alpha = 50

# Use the midpoint of each interval
omega_edges = np.linspace(omega_min, omega_max, n_validation_omega + 1)
validation_omegas = 0.5 * (omega_edges[:-1] + omega_edges[1:])

alpha_edges = np.linspace(alpha_min, alpha_max, n_validation_alpha + 1)
validation_alphas = 0.5 * (alpha_edges[:-1] + alpha_edges[1:])

validation_pairs = []

for omega in validation_omegas:
    for alpha in validation_alphas:
        validation_pairs.append([omega, alpha])

validation_pairs = np.asarray(validation_pairs)

true_status = []
true_next_omega = np.full(len(validation_pairs),np.nan)

for k in range(len(validation_pairs)):

    omega = validation_pairs[k, 0]
    alpha = validation_pairs[k, 1]

    status, result = (
        simulate.simulate_step(
            omega,
            alpha,
            theta_grid,
            angular_vel_grid,
            roa,
            params
        )
    )

    true_status.append(status)

    if status == "poincare":
        true_next_omega[k] = result

true_status = np.asarray(true_status, dtype=str)

def lookup_prediction(omega, alpha, lookup_data):

    omega_grid = (lookup_data["omega_grid"])
    alpha_grid = (lookup_data["alpha_grid"])
    next_omega_table = (lookup_data["next_omega_table"])
    reaches_roa = (lookup_data["reaches_roa"])
    failed_table = (lookup_data["failed_table"])
    already_in_roa = (lookup_data["already_in_roa"])

    if (
        omega < omega_grid[0] or
        omega > omega_grid[-1] or
        alpha < alpha_grid[0] or
        alpha > alpha_grid[-1]
    ):
        return ("outside", np.nan)

    i = np.argmin( np.abs(omega_grid - omega))
    j = np.argmin(np.abs(alpha_grid - alpha))

    if already_in_roa[i]:
        return ("already in roa", np.nan)

    if reaches_roa[i, j]:
        return ("roa", np.nan)

    if not np.isnan(next_omega_table[i, j]):
        return ("poincare", next_omega_table[i, j])

    if failed_table[i, j]:
        return ("failed", np.nan)

    return ("unknown", np.nan)


summary = []
all_predictions = {}

for resolution in resolutions:

    lookup_file = (resolution_dir / f"lookup_{resolution}x{resolution}.npz")
    lookup_data = np.load(lookup_file)

    omega_grid = (lookup_data["omega_grid"])
    alpha_grid = (lookup_data["alpha_grid"])

    delta_omega = float(lookup_data["delta_omega"])
    delta_alpha = float(lookup_data["delta_alpha"])

    predicted_status = []
    predicted_next_omega = np.full(len(validation_pairs), np.nan)

    omega_lookup_error = np.zeros(len(validation_pairs))
    alpha_lookup_error = np.zeros(len(validation_pairs))

    for k in range(len(validation_pairs)):

        omega = validation_pairs[k, 0]
        alpha = validation_pairs[k, 1]

        i = np.argmin(np.abs(omega_grid - omega))
        j = np.argmin(np.abs(alpha_grid - alpha))

        omega_lookup_error[k] = abs(omega - omega_grid[i])
        alpha_lookup_error[k] = abs(alpha - alpha_grid[j])

        status, next_omega = (lookup_prediction(omega, alpha, lookup_data))
        predicted_status.append(status)
        predicted_next_omega[k] = (next_omega)

    predicted_status = np.asarray(predicted_status, dtype=str)

    classification_mismatch = (predicted_status != true_status)
    classification_mismatch_count = int(np.sum(classification_mismatch))
    classification_mismatch_rate = (classification_mismatch_count / len(validation_pairs))


    comparable = ((true_status == "poincare") & (predicted_status == "poincare"))

    if np.any(comparable):

        transition_error = np.abs(predicted_next_omega[comparable] - true_next_omega[comparable])
        mean_transition_error = np.mean(transition_error)
        max_transition_error = np.max(transition_error)
        rms_transition_error = np.sqrt(np.mean(transition_error ** 2))

        number_comparable = np.sum(comparable)


    else:
        mean_transition_error = np.nan
        max_transition_error = np.nan
        rms_transition_error = np.nan
        number_comparable = 0


    max_omega_lookup_error = np.max(omega_lookup_error)
    mean_omega_lookup_error = np.mean(omega_lookup_error)
    max_alpha_lookup_error = np.max(alpha_lookup_error)
    mean_alpha_lookup_error = np.mean(alpha_lookup_error)


    summary.append(
        [
            resolution,

            delta_omega,
            delta_alpha,

            max_omega_lookup_error,
            mean_omega_lookup_error,

            max_alpha_lookup_error,
            mean_alpha_lookup_error,

            classification_mismatch_count,
            classification_mismatch_rate,

            number_comparable,

            mean_transition_error,
            max_transition_error,
            rms_transition_error,
        ]
    )


    all_predictions[resolution] = {
        "status": predicted_status,
        "next_omega": predicted_next_omega
    }


# Compare grid refinment from each step

refinement_results = []

for k in range(len(resolutions) - 1):
    coarse_resolution = (resolutions[k])
    fine_resolution = (resolutions[k + 1])
    
    coarse_status = (all_predictions[coarse_resolution]["status"])
    fine_status = (all_predictions[fine_resolution]["status"])

    coarse_next_omega = (all_predictions[coarse_resolution]["next_omega"])
    fine_next_omega = (all_predictions[fine_resolution]["next_omega"])

    status_disagreements = int(np.sum(coarse_status != fine_status))

    comparable = ((coarse_status == "poincare") & (fine_status == "poincare"))
    if np.any(comparable):
        transition_change = np.abs(coarse_next_omega[comparable] - fine_next_omega[comparable])
        mean_transition_change = np.mean(transition_change)
        max_transition_change = np.max(transition_change)

    else:
        mean_transition_change = np.nan
        max_transition_change = np.nan

    refinement_results.append(
        [
            coarse_resolution,
            fine_resolution,
            status_disagreements,
            mean_transition_change,
            max_transition_change,
        ]
    )


summary_array = np.asarray(summary,dtype=float)

np.savetxt(
    resolution_dir
    / "grid_resolution_summary.csv",

    summary_array,

    delimiter=",",

    header=(
        "resolution,"
        "delta_omega,"
        "delta_alpha,"
        "max_omega_lookup_error,"
        "mean_omega_lookup_error,"
        "max_alpha_lookup_error,"
        "mean_alpha_lookup_error,"
        "classification_mismatch_count,"
        "classification_mismatch_rate,"
        "number_comparable,"
        "mean_transition_error,"
        "max_transition_error,"
        "rms_transition_error"
    ),

    comments=""
)


refinement_array = np.asarray(refinement_results, dtype=float)
resolution_values = (summary_array[:, 0])

mean_transition_errors = (summary_array[:, 10])
max_transition_errors = (summary_array[:, 11])
rms_transition_errors = (summary_array[:, 12])

plt.figure(figsize=(8, 5))

plt.plot(
    resolution_values,
    mean_transition_errors,
    marker="o",
    label="Mean Error"
)

plt.plot(
    resolution_values,
    rms_transition_errors,
    marker="o",
    label="RMS Error"
)

plt.plot(
    resolution_values,
    max_transition_errors,
    marker="o",
    label="Maximum Error"
)

plt.xlabel("Lookup Table Resolution")
plt.ylabel(r"Error in $\dot{\theta}_{k+1}$ [rad/s]")

plt.title("Poincaré Transition-Map Error vs. Grid Resolution")

plt.xticks(resolution_values)

plt.grid()
plt.legend()
plt.tight_layout()

plt.show()

refinement_labels = [
    f"{int(row[0])}"
    f"\N{RIGHTWARDS ARROW}"
    f"{int(row[1])}"

    for row in refinement_results
]

classification_changes = (refinement_array[:, 2])

plt.figure(figsize=(8, 5))

plt.plot(
    refinement_labels,
    classification_changes,
    marker="o"
)

plt.xlabel("Successive Grid Refinement")
plt.ylabel("Number of Transition Classification Changes")
plt.title("Sensitivity of Transition Classification to Grid Refinement")

plt.grid()
plt.tight_layout()
plt.show()