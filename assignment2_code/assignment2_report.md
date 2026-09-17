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
To stabilize the inverted pendulum in the upright equilibrium we can configure a control input u:

### Poincare Section


### Control as a lookup table