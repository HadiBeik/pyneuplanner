import numpy as np
import math
from keras.initializers import normal, identity, uniform
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 10
HIDDEN2_UNITS = 10


class CriticNetwork(object):

    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.actionDim = action_size
        K.set_session(sess)
        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        print("Now we build the model")
        action_input = Input(shape=[action_dim], name='action_input')
        observation_input = Input(shape=[state_size], name='observation_input')
        # flattened_observation = Flatten()(observation_input)
        x = Dense(400)(observation_input)
        x = Activation('relu')(x)
        x = concatenate([x, action_input])
        x = Dense(300)(x)
        x = Activation('relu')(x)
        # x = Dense(300)(x)
        # x = Activation('relu')(x)
        # x = Dense(self.actionDim)(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(input=[observation_input, action_input], output=x)
        # critic = Model(input=[observation_input,action_input], output=x)
        print(critic.summary())
        A = action_input
        S = observation_input
        adam = Adam(lr=self.LEARNING_RATE)
        critic.compile(loss='mse', optimizer=adam)
        return critic, A, S
