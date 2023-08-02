import gym
from source.wrapper.skipper import make_env

env = make_env('CartPole-v1', 1, 3, False, render_mode='none')
env.reset()

for _ in range(5):
    done = False
    trunc = False
    while not done and not trunc:
        ap = env.action_space.sample()
        print(ap)
        s, r, done, trunc, i = env.step(ap)
        print(i)
        print(r)
    env.reset()
