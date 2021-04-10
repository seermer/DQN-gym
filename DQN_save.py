import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gc
import os
import shutil


class DQN:
    def __init__(self,
                 num_actions,
                 state_size,
                 target_replace_step=2048,
                 discount=.9,
                 epsilon=.9995,
                 mem_size=8192,
                 optimizer=tf.optimizers.Adam(learning_rate=.1),
                 batch_size=256,
                 epsilon_decrement=.99975,
                 seed=None
                 ):
        if seed:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        self.n_actions = num_actions
        self.dim = len(state_size)
        if self.dim > 3:
            raise ValueError("special input requires other processing")
        self.n_states = state_size[0] if self.dim == 1 else state_size
        self.target_replace_step = target_replace_step
        self.discount = discount
        self.epsilon = epsilon
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.epsilon_decrement = epsilon_decrement

        self.step = 0
        self.learn_step = 0
        self.memory = np.zeros((mem_size, 3))
        # (state, next_state), action, reward, done

        self.train_model = self._get_model()
        self.target_model = tf.keras.models.clone_model(self.train_model)
        self.target_model.set_weights(self.train_model.get_weights())
        self.train_model.compile(optimizer=optimizer,
                                 loss=tf.losses.mean_squared_error)
        self.train_model.trainable = True
        self.target_model.trainable = False
        if not os.path.exists("./replays"):
            os.mkdir("./replays")
        if len(os.listdir("./replays/")) != 0:
            ans = input("replay folder is not empty, clear it? (y/n)")
            if "y" in ans.lower():
                shutil.rmtree("./replays")
                os.mkdir("./replays")
                print("cleared")
            else:
                exit(0)

    def _get_model(self):
        if self.dim == 1:
            inp = layers.Input((self.n_states,))
            x = layers.Dense(32)(inp)

        else:
            inp = layers.Input(self.n_states)
            pre = tf.keras.applications.MobileNetV3Small(input_tensor=inp, minimalistic=True, include_top=False,
                                                         dropout_rate=0.01, pooling="avg")
            x = layers.Dense(128)(pre.output)

        x = layers.Activation(tf.keras.activations.elu)(x)
        x = layers.Dense(self.n_actions)(x)

        model = tf.keras.Model(inp, x)

        return model

    def store_mem(self, s, a, r, s_, done):
        findex = self.step % self.mem_size
        save = np.array([s, s_])
        np.save("./replays/" + str(findex), save)
        self.memory[findex, :] = [a, r, 0 if done else 1]
        self.step += 1

    def choose_action(self, s):
        s = np.array([s])
        selector = np.random.choice([0, 1], p=(self.epsilon, 1 - self.epsilon))
        if self.epsilon > .04 and self.step > 300:
            self.epsilon *= self.epsilon_decrement
        if selector == 0:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.train_model.predict(s))

    def _replace_target(self):
        self.target_model.set_weights(self.train_model.get_weights())
        print("target model weights replaced")

    def learn(self):
        if self.learn_step % self.target_replace_step == 1:
            self._replace_target()
        if self.step <= self.batch_size:
            return

        samples = self.memory[:min(self.step, self.mem_size)]
        sample_index = np.random.randint(0, len(samples), self.batch_size)
        samples = samples[sample_index, :]

        sslist1 = []
        sslist2 = []
        for index in sample_index:
            temp = np.load("./replays/" + str(index) + ".npy")
            sslist1.append(temp[0])
            sslist2.append(temp[1])

        s = np.array(sslist1)
        s_ = np.array(sslist2)
        a = samples[:, 0]
        r = samples[:, 1]
        done = samples[:, 2]

        del temp, sslist1, sslist2, samples, sample_index
        gc.collect()

        Q_pred = self.train_model.predict(s)
        Q_next_target = self.target_model.predict(s_)

        for i in range(len(done)):
            new_Q = r[i] + self.discount * np.max(Q_next_target[i]) * done[i]
            Q_pred[i, int(a[i])] = new_Q

        self.train_model.train_on_batch(x=s, y=Q_pred)
        self.learn_step += 1
