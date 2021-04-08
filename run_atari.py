import gym
from DQN import DQN

ENV = gym.make("Pong-v0")
EPISODES = 2000

dqn = DQN(ENV.action_space.n, ENV.observation_space.shape, mem_size=2048, batch_size=64)

for episode in range(EPISODES):
    s = ENV.reset()
    acc_r = 0
    while True:
        ENV.render()

        a = dqn.choose_action(s)
        s_, r, done, info = ENV.step(a)
        acc_r += r
        dqn.store_mem(s, a, r, s_, done)
        dqn.learn()

        if done:
            print("episode:", episode, " reward:", round(acc_r, 4))
            break

        s = s_
