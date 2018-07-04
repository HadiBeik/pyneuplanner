import numpy as np
import math
from keras.initializers import normal, identity, uniform
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation,Convolution1D
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.stateDim=8
        self.actionDim=2
        K.set_session(sess)
        #  create actor network
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        #  create actor target network
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        #  actor gradient init
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        #
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        #  init gradients
        grads = zip(self.params_grad, self.weights)
        #  finding the pick
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        #  Loading weights of actor network
        actor_weights = self.model.get_weights()
        #  loading weights of actor target network
        actor_target_weights = self.target_model.get_weights()
        #  update actor target network by actor network weights (TAU * act) +  (1-TAU * act_tar)
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        #  update actor target model
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        print("Now we build the model")
        observation_input = Input(shape=[state_size], name='observation_input')
        x = Dense(400)(observation_input)
        x = Activation('relu')(x)
        x = Dense(300)(x)
        x = Activation('relu')(x)
        # x=Dense(300)(x)
        # x=Activation('relu')(x)
        x = Dense(self.actionDim)(x)
        x = Activation('sigmoid')(x)
        actor = Model(input=observation_input, output=x)
        print(actor.summary())
        S = observation_input
        return actor, actor.trainable_weights, S
