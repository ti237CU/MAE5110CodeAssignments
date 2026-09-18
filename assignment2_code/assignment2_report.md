# Assignment 2
Thomas Istvan (ti237)
\
09/16/2026

### Initial Sketching

### Stabilizing with Feedback Linearization
Using the inverted pendulum dynamics with actuation we get the equation:
$$
\ddot{\theta} = \frac{g}{l}sin(\theta) + \frac{\tau}{ml^2}
$$
Applying feedback linearization to cancel out the non-linear term and introducing a desired control input u:
$$
\ddot{\theta} = u
$$
$$
u = \frac{g}{l}sin(\theta) + \frac{\tau}{ml^2}
$$
Isolating for the ankle torque:
$$
\tau = -ml^2\frac{g}{l}sin(\theta) - ml^2u
$$
$$
\tau = -mglsin(\theta) - ml^2u
$$
To stabilize the inverted pendulum in the upright equilibrium we can configure a control input u to add artifical damping and friction:
$$
u = -ml^2(K_d\dot{\theta} + K_p\theta)
$$
After feedback linearization, the dynamics are:
$$
\ddot{\theta} + K_d\dot{\theta} + K_p\theta = 0
$$
$$
\ddot{\theta} + 2\zeta{\omega_n}\dot{\theta} + {\omega_n}^2\theta = 0
$$
Relating the coefficents we can see that:
$$
K_d = 2\zeta{\omega_n}
$$
$$
K_p = {\omega_n}^2
$$

### Region of Attraction
To produce my region of attraction my small grid search was based on the set contstrains of the angle of attack and the incline, which would allow for a max and minimum value of theta. Then using those values of theta, we could calculate a minimum and maximum value for the angular velocity to be used in the grid.

### Poincare Section

### Control as a lookup table