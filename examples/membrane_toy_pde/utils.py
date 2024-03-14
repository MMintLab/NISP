import jax.numpy as jnp
import scipy.io

import deepxde as dde
from bubble_tools.bubble_tools.bubble_ellipsoid_tools import load_ellipse




def get_dataset(N=100000):

    a, b = 0.04, 0.04
    E, P, t, nu = 1e9, 2757, 0.0003, 0.5
    ellipse_points = load_ellipse(2*a, 2*b, num_points=10000)
    
    geom = dde.geometry.Ellipse([0,0], a, b)
    train_points =  geom.uniform_points(N)


    coords = jnp.array(train_points)
    wall_coords = jnp.array(ellipse_points)

    E = jnp.array(E)
    P = jnp.array(P)
    t = jnp.array(t)
    nu = jnp.array(nu)

    a = jnp.array(a)
    b = jnp.array(b)
    
    return (
        coords,
        wall_coords,
        E,
        P,
        t, 
        nu,
        a, 
        b 
    )
    
