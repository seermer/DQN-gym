import gym
from DQN import DQN

ENV = gym.make("CartPole-v1")
EPISODES = 1052

dqn = DQN(ENV.action_space.n, ENV.observation_space.shape)

for episode in range(EPISODES):
    s = ENV.reset()
    acc_r = 0
    if episode == 512:
        dqn.epsilon = .005
    while True:
        ENV.render()

        a = dqn.choose_action(s)
        s_, r, done, _ = ENV.step(a)
        acc_r += r

        r = r / ((abs(s_[2]) * 10) if abs(s[2]) > .1 else 1)  # custom reward according to angle

        dqn.store_mem(s, a, r, s_, done)
        dqn.learn()

        if done:
            print("episode:", episode, " reward:", acc_r)
            break

        s = s_


