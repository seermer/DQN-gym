import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class DQN:
    def __init__(self,
                 num_actions,
                 state_size,
                 target_replace_step=2048,
                 discount=.9,
                 epsilon=1,
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
        self.memory = np.zeros((mem_size, 3 + 2 * (self.n_states if self.dim == 1 else np.prod(self.n_states))))
        # state, action, reward, next_state

        self.train_model = self._get_model()
        self.target_model = tf.keras.models.clone_model(self.train_model)
        self.target_model.set_weights(self.train_model.get_weights())
        self.train_model.compile(optimizer=optimizer,
                                 loss=tf.losses.mean_squared_error)
        self.train_model.trainable = True
        self.target_model.trainable = False

    def _get_model(self):
        if self.dim == 1:
            inp = layers.Input((self.n_states,))
            x = layers.Dense(32)(inp)

        else:
            inp = layers.Input(self.n_states)
            x = layers.Conv2D(32, (3, 3))(inp)
            x = layers.Activation(tf.keras.activations.elu)(x)
            x = layers.Conv2D(64, (3, 3))(x)
            x = layers.Activation(tf.keras.activations.elu)(x)
            x = layers.Conv2D(128, (3, 3))(x)
            x = layers.GlobalAveragePooling2D()(x)

        x = layers.Activation(tf.keras.activations.elu)(x)

        x = layers.Dense(self.n_actions)(x)

        model = tf.keras.Model(inp, x)

        return model

    def store_mem(self, s, a, r, s_, done):
        if self.dim != 1:
            s = s.flatten()
            s_ = s_.flatten()
        self.memory[self.step % self.mem_size, :] = [*s, *s_, a, r, 0 if done else 1]
        self.step += 1

    def choose_action(self, s):
        s = np.array([s])
        selector = np.random.choice([0, 1], p=(self.epsilon, 1 - self.epsilon))
        if self.epsilon > .04 and self.step > 500:
            self.epsilon *= self.epsilon_decrement
        if selector == 0:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.train_model.predict(s))

    def _replace_target(self):
        self.target_model.set_weights(self.train_model.get_weights())
        print("target model weights replaced")

    def learn(self):
        if self.step <= self.batch_size:
            return

        samples = self.memory[:min(self.step, self.mem_size)]
        samples = samples[np.random.randint(0, len(samples), self.batch_size), :]

        slicer = self.n_states if self.dim == 1 else np.prod(self.n_states)
        s = samples[:, :slicer]
        s_ = samples[:, slicer:slicer + slicer]
        a = samples[:, -3]
        r = samples[:, -2]
        done = samples[:, -1]

        if self.dim != 1:
            s = np.reshape(s, (self.batch_size,) + self.n_states)
            s_ = np.reshape(s_, (self.batch_size,) + self.n_states)

        Q_pred = self.train_model.predict(s)
        Q_next_target = self.target_model.predict(s_)

        for i in range(len(done)):
            new_Q = r[i] + self.discount * np.max(Q_next_target[i]) * done[i]
            Q_pred[i, int(a[i])] = new_Q

        self.train_model.fit(x=s, y=Q_pred, shuffle=False, verbose=0, batch_size=self.batch_size)

        if self.step % self.target_replace_step == 1:
            self._replace_target()
