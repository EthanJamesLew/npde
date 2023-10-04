import pandas as pd
import numpy as np

from typing import Tuple

from npde_helper import build_model, fit_model, save_model, load_model

import tensorflow as tf
sess = tf.InteractiveSession()

# dataframe schema names
states_name = [f"X{i}" for i in range(1, 7)]
tid_name = "id"
t_name = "t"

def get_trajs(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """get trajectory information from a dataframe in a format accepted by npode"""
    states = []
    times = []
    trajs = df.groupby(tid_name)
    for tid, traj in trajs:
        t = traj[t_name]
        s = traj[states_name]
        states.append(s)
        times.append(t)
    return np.array(times), np.array(states)


df = pd.read_csv("./data/lorenz96_6_no_noise/Lorenz_96_6_no_noise_trajs.csv")
t, Y = get_trajs(df)
N = 3
print(t.shape, Y.shape)
t, Y = t[:10, :], Y[:10, :, :N]
npde = build_model(sess, t, Y, model='ode', sf0=1.0, ell0=np.ones(N), W=6, ktype="id")
npde = fit_model(sess, npde, t, Y, num_iter=500, print_every=50, eta=0.02, plot_=True)


