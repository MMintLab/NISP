import os
import numpy as np
import open3d as o3d
import jax.numpy as jnp

import scipy.signal
import matplotlib.pyplot as plt
from scipy.integrate import odeint

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Original problem : https://github.com/pydy/pydy_examples/blob/master/npendulum/n-pendulum-control.ipynb and https://x-engineer.org/quarter-car-suspension-transfer-function/
# from __future__ import division, print_function
import sympy as sm
import sympy.physics.mechanics as me
from numpy.linalg import matrix_rank
from scipy.linalg import solve_continuous_are
from numpy.linalg import solve

n = 2
arm_length = 1.0 / n  # The maximum length of the pendulum is 1 meter
bob_mass = 0.01 / n  # The maximum mass of the bobs is 10 grams

q = me.dynamicsymbols("q:{}".format(n + 1))  # Generalized coordinates
u = me.dynamicsymbols("u:{}".format(n + 1))  # Generalized speeds
f = me.dynamicsymbols("f")  # Force applied to the cart

m = sm.symbols("m:{}".format(n + 1))  # Mass of each bob
l = sm.symbols("l:{}".format(n))  # Length of each link
g, t = sm.symbols("g t")  # Gravity and time
parameters = [g, m[0]]  # Parameter definitions starting with gravity and the first bob
parameter_vals = [9.81, 0.01 / n]  # Numerical values for the first two
for i in range(n):  # Then each mass and length
    parameters += [l[i], m[i + 1]]
    parameter_vals += [arm_length, bob_mass]

I = me.ReferenceFrame("I")  # Inertial reference frame
O = me.Point("O")  # Origin point
O.set_vel(I, 0)  # Origin's velocity is zero

P0 = me.Point("P0")  # Hinge point of top link
P0.set_pos(O, q[0] * I.x)  # Set the position of P0
P0.set_vel(I, u[0] * I.x)  # Set the velocity of P0
Pa0 = me.Particle("Pa0", P0, m[0])  # Define a particle at P0

frames = [I]  # List to hold the n + 1 frames
points = [P0]  # List to hold the n + 1 points
particles = [Pa0]  # List to hold the n + 1 particles
forces = [
    (P0, f * I.x - m[0] * g * I.y)
]  # List to hold the n + 1 applied forces, including the input force, f
kindiffs = [q[0].diff(t) - u[0]]  # List to hold kinematic ODE's

for i in range(n):
    Bi = I.orientnew("B" + str(i), "Axis", [q[i + 1], I.z])  # Create a new frame
    Bi.set_ang_vel(I, u[i + 1] * I.z)  # Set angular velocity
    frames.append(Bi)  # Add it to the frames list

    Pi = points[-1].locatenew("P" + str(i + 1), l[i] * Bi.x)  # Create a new point
    Pi.v2pt_theory(points[-1], I, Bi)  # Set the velocity
    points.append(Pi)  # Add it to the points list

    Pai = me.Particle("Pa" + str(i + 1), Pi, m[i + 1])  # Create a new particle
    particles.append(Pai)  # Add it to the particles list

    forces.append((Pi, -m[i + 1] * g * I.y))  # Set the force applied at the point

    kindiffs.append(
        q[i + 1].diff(t) - u[i + 1]
    )  # Define the kinematic ODE:  dq_i / dt - u_i = 0


dynamic = q + u  # Make a list of the states
dynamic.append(f)  # Add the input force

kane = me.KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)  # Initialize the object
fr, frstar = kane.kanes_equations(particles, forces)
M_func = sm.lambdify(
    dynamic + parameters, kane.mass_matrix_full
)  # Create a callable function to evaluate the mass matrix
f_func = sm.lambdify(dynamic + parameters, kane.forcing_full)

# Generate EoM's fr + frstar = 0
sm.trigsimp(kane.mass_matrix)
me.find_dynamicsymbols(kane.mass_matrix)
sm.trigsimp(kane.forcing)
me.find_dynamicsymbols(kane.forcing)

equilibrium_point = [sm.S(0)] + [sm.pi / 2] * (len(q) - 1) + [sm.S(0)] * len(u)
equilibrium_dict = dict(zip(q + u, equilibrium_point))
M, F_A, F_B, r = kane.linearize(new_method=True, op_point=equilibrium_dict)
parameter_dict = dict(zip(parameters, parameter_vals))

M_num = sm.matrix2numpy(M.subs(parameter_dict), dtype=float)
F_A_num = sm.matrix2numpy(F_A.subs(parameter_dict), dtype=float)
F_B_num = sm.matrix2numpy(F_B.subs(parameter_dict), dtype=float)

A = np.linalg.solve(M_num, F_A_num)
B = np.linalg.solve(M_num, F_B_num)

equilibrium_point = np.asarray([x.evalf() for x in equilibrium_point], dtype=float)


def controllable(a, b):
    """Returns true if the system is controllable and false if not.
    Parameters
    ----------
    a : array_like, shape(n,n)
        The state matrix.
    b : array_like, shape(n,r)
        The input matrix.
    Returns
    -------
    controllable : boolean
    """
    a = np.matrix(a)
    b = np.matrix(b)
    n = a.shape[0]
    controllability_matrix = []
    for i in range(n):
        controllability_matrix.append(a**i * b)
    controllability_matrix = np.hstack(controllability_matrix)

    return np.linalg.matrix_rank(controllability_matrix) == n


def right_hand_side(x, t, args):
    """Returns the derivatives of the states.

    Parameters
    ----------
    x : ndarray, shape(2 * (n + 1))
        The current state vector.
    t : float
        The current time.
    args : ndarray
        The constants.

    Returns
    -------
    dx : ndarray, shape(2 * (n + 1))
        The derivative of the state.

    """
    r = np.dot(K, equilibrium_point - x)  # The controller
    arguments = np.hstack((x, r, args))  # States, input, and parameters
    dx = np.array(
        solve(M_func(*arguments), f_func(*arguments))  # Solving for the derivatives
    ).T[0]

    return dx


def right_hand_side_custom(x, t, args, r_fn):
    """Returns the derivatives of the states.

    Parameters
    ----------
    x : ndarray, shape(2 * (n + 1))
        The current state vector.
    t : float
        The current time.
    args : ndarray
        The constants.

    Returns
    -------
    dx : ndarray, shape(2 * (n + 1))
        The derivative of the state.

    """
    r_est = np.array(r_fn(jnp.array([t])))
    r_gt = np.dot(K, equilibrium_point - x)  # The controller
    print(t, r_est - r_gt)

    if t > 0:
        r = r_est
    else:
        r = r_gt  # The controller

    arguments = np.hstack((x, r, args))  # States, input, and parameters

    dx = np.array(
        solve(M_func(*arguments), f_func(*arguments))  # Solving for the derivatives
    ).T[0]

    return dx


Q = np.eye(A.shape[0])
R = np.eye(B.shape[1]) * 150
S = solve_continuous_are(A, B, Q, R)
K = np.dot(np.dot(np.linalg.inv(R), B.T), S)


from matplotlib import animation
from matplotlib.patches import Rectangle


def animate_pendulum(t, states, length, filename=None):
    """Animates the n-pendulum and optionally saves it to file.

    Parameters
    ----------
    t : ndarray, shape(m)
        Time array.
    states: ndarray, shape(m,p)
        State time history.
    length: float
        The length of the pendulum links.
    filename: string or None, optional
        If true a movie file will be saved of the animation. This may take some time.

    Returns
    -------
    fig : matplotlib.Figure
        The figure.
    anim : matplotlib.FuncAnimation
        The animation.

    """
    # the number of pendulum bobs
    numpoints = states.shape[1] // 2

    # first set up the figure, the axis, and the plot elements we want to animate
    fig = plt.figure()

    # some dimesions
    cart_width = 0.4
    cart_height = 0.2

    # set the limits based on the motion
    xmin = np.around(states[:, 0].min() - cart_width / 2.0, 1)
    xmax = np.around(states[:, 0].max() + cart_width / 2.0, 1)

    # create the axes
    ax = plt.axes(xlim=(xmin, xmax), ylim=(-1.1, 1.1), aspect="equal")

    # display the current time
    time_text = ax.text(0.04, 0.9, "", transform=ax.transAxes)

    # create a rectangular cart
    rect = Rectangle(
        [states[0, 0] - cart_width / 2.0, -cart_height / 2],
        cart_width,
        cart_height,
        fill=True,
        color="red",
        ec="black",
    )
    ax.add_patch(rect)

    # blank line for the pendulum
    (line,) = ax.plot([], [], lw=2, marker="o", markersize=6)

    # initialization function: plot the background of each frame
    def init():
        time_text.set_text("")
        rect.set_xy((0.0, 0.0))
        line.set_data([], [])
        return (
            time_text,
            rect,
            line,
        )

    # animation function: update the objects
    def animate(i):
        time_text.set_text("time = {:2.2f}".format(t[i]))
        rect.set_xy((states[i, 0] - cart_width / 2.0, -cart_height / 2))
        x = np.hstack((states[i, 0], np.zeros((numpoints - 1))))
        y = np.zeros((numpoints))
        for j in np.arange(1, numpoints):
            x[j] = x[j - 1] + length * np.cos(states[i, j])
            y[j] = y[j - 1] + length * np.sin(states[i, j])
        line.set_data(x, y)
        return (
            time_text,
            rect,
            line,
        )

    # call the animator function
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(t),
        init_func=init,
        interval=t[-1] / len(t) * 1000,
        blit=True,
        repeat=False,
    )

    # save the animation if a filename is given
    if filename is not None:
        anim.save(filename, fps=30, codec="libx264")


def get_dataset(N=1000, a=0, b=0, c=0, d=0, noise=0):
    x0 = np.hstack([0, np.pi / 2 + a, np.pi / 2 + b, 0, c, d])
    t = np.linspace(0.0, 3.0, num=N)
    t_dense = np.linspace(0.0, 3.0, num=4 * N)
    x_without_noise = odeint(right_hand_side, x0, t, args=(parameter_vals,))
    u_control = [np.dot(K, equilibrium_point - x_i) for x_i in x_without_noise]

    fig = plt.figure()
    plt.plot(t[1:], u_control[1:])
    plt.xlabel("time [s]")
    plt.ylabel("Control Input [m]")
    plt.legend(["analytical", "input"])
    plt.show()
    plt.savefig(f"result/forcing_term_{a}_{b}_{c}_{d}.png")
    plt.close()

    # fig = plt.figure()
    # plt.plot(t, x[:, :x.shape[1] // 2])
    # plt.xlabel("time [s]")
    # plt.ylabel("State [m] or rad")
    # plt.legend(["analytical", "input"])
    # plt.show()
    # plt.savefig(f"result/states_{a}_{b}_{c}_{d}.png")
    # plt.close()

    g = jnp.array(parameter_vals[0])
    m0 = jnp.array(parameter_vals[1])
    l = jnp.array(parameter_vals[2])
    m1 = jnp.array(parameter_vals[3])
    m2 = jnp.array(parameter_vals[5])
    t_dense = jnp.array(t_dense)[..., jnp.newaxis]

    np.random.seed(42)
    # Noisy Data
    x = np.tile(x_without_noise, (5, 1))
    t = np.tile(t, 5)

    # Noise Injection
    x = x + np.random.normal(0, noise, size=x.shape)
    q0 = jnp.array(x[:, 0])[..., jnp.newaxis]
    q1 = jnp.array(x[:, 1])[..., jnp.newaxis]
    q2 = jnp.array(x[:, 2])[..., jnp.newaxis]
    t = jnp.array(t)[..., jnp.newaxis]

    u_control = jnp.array(u_control)
    L = 1

    return (L, t, t_dense, g, m0, m1, m2, l, q0, q1, q2, u_control, x_without_noise)


if __name__ == "__main__":
    get_dataset()
