import thread
import numpy as np
import Queue
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import math
from non_physical_simulation import simulator
import sys
import time
import pickle
from keras import backend as K
from emailNotification import emailNotificationClass
import pypot.dynamixel


class DdpgAgent:

    def __init__(self, Run_Num, Dynamic):

        self.Memory_Leak = 0
        self.email = emailNotificationClass()
        self.run_Num = Run_Num
        print(str(Run_Num))
        print(str(Dynamic))
        self.DYNAMIC_SIZE_TARGET = Dynamic
        self.TRAIN = True
        ################################################################
        uni = True
        self.env = simulator()
        self.MEMORY_SIZE = 1000000

        self.CONTINUE_LEARNING = False
        ###############################################################
        self.ACTOR_LEARNING_RATE = 0.0001  # Actor learning rate
        self.CRITIC_LEARNING_RATE = 0.0002  # Critic learning rate

        self.MAIN_MEMORY_SAVE_SIZE_TH = 5
        self.BATCH_SIZE = 32 if self.TRAIN else 32
        self.MIN_EXPLORATION_RATE = 0.05
        self.ALPHA = 0.99
        self.Current_Story = 0
        self.TNH = 0.01
        self.target_size = 10  # initial value of target size
        self.EPSILON = 0.3 if self.TRAIN else 0.0  # initial value of epsilon
        self.LimitOutput = False
        self.LimitValue = 5 * math.pi / 180  # 5 degrees
        self.a_d = 2  # of actuated joints
        self.s_d = 7  # num of features in state
        self.STORY_N = 10 if self.TRAIN else 1  # max number of Stories
        self.FINE_TUNE_N = 10  # number of episodes after learning
        self.EXPLORE_N = 200  # defines the exploration decay rate
        self.Episode_N = 500 if self.TRAIN else 1  # max episode number in Story
        self.steps_max = 500  # max iteration number in each episode
        self.step = 0
        self.check_queue_size = 10
        self.TARGET_MINIMUM_SIZE = 10
        self.TARGET_MAXIMUM_SIZE = 100 if self.DYNAMIC_SIZE_TARGET else 10
        self.TERMINAL_WRITE = True
        self.terminal = False
        self.buffer = ReplayBuffer(self.MEMORY_SIZE)  # Create replay buffer
        self.vis_buffer = ReplayBuffer(360 * 360 * 9)
        self.gpu_setup()
        self.target_X = 0  # cartesian position of target  (Y axis)
        self.target_Y = 0  # cartesian position of target (X Axis)
        self.angle_buffer = np.zeros((2000, 2))  # To draw the trajectory of the movement

        # self.target_index = 0  # every target has an index that we have save it in the file with that index
        self.Visualizer = False
        self.Target_Vector_Orientation = 0  # Theta of target in Polar space
        self.ACCELERATION_RATE = 1  # number of pixel that we add each time
        self.DECCELERATION_RATE = 1  # number of pixel that we subtract each time
        self.Target_Vector_Size = 200  # R of Target in Polar space
        self.train_data = np.zeros((10, 2))
        self.test_data = np.zeros((10, 2))
        # self.generate_target(30, 30)
        self.Trajectory_Visualization = False
        no_data_existed = False
        # if no_data_existed:
        #     self.generate_all_data()
        # self.random_indexes = pickle.load(open('Points.p', "rb"))
        self.actor = ActorNetwork(self.sess, self.s_d, self.a_d, self.BATCH_SIZE, self.TNH, self.ACTOR_LEARNING_RATE)
        self.critic = CriticNetwork(self.sess, self.s_d, self.a_d, self.BATCH_SIZE, self.TNH, self.CRITIC_LEARNING_RATE)
        self.success_queue = Queue.Queue(maxsize=self.check_queue_size)  # History of suc and unsuc trials
        # while not self.vision.detecting_completed:
        #     self.vision.marker_detector(True)
        #     self.actual_target = self.vision.target_position

    def reset_everything(self):
        self.angle_buffer = np.zeros((2000, 2))
        # self.env = Simulator()
        self.Trajectory_Visualization = False
        self.success_queue = Queue.Queue(maxsize=self.check_queue_size)  # History of suc and unsuc trials

    # Call this function just one time to generate visualization data
    def generate_all_data(self):
        # for i in range(0, len(self.test_data), 1):
        #     self.generate_visualization_data(self.test_data[i][0], self.test_data[i][1])
        #     pickle.dump(self.states_vis, open(self.media_address + 'Reservoir/Test/State' + str(i) + '.p', "wb"))
        #     pickle.dump(self.actions_vis, open(self.media_address + 'Reservoir/Test/Action' + str(i) + '.p', "wb"))
        #     print ('Test: ' + str(i) + '/' + str(len(self.test_data)))
        for i in range(0, len(self.train_data), 1):
            self.generate_visualization_data(self.train_data[i][0], self.train_data[i][1])
            pickle.dump(self.states_vis, open(self.media_address + 'Reservoir/Train/State' + str(i) + '.p', "wb"))
            pickle.dump(self.actions_vis, open(self.media_address + 'Reservoir/Train/Action' + str(i) + '.p', "wb"))
            print ('Train: ' + str(i) + '/' + str(len(self.train_data)))

    # Transform Targets
    # def generate_target(self, angle_res, distance_res):
    #     h, w = self.env.tracker.target_position.shape
    #     for i in range(h):
    #         rho, phi = self.cart2pol(self.env.tracker.target_position[i][0], self.env.tracker.target_position[i][1])
    #         self.Target_Vector_Orientation = phi
    #         self.Target_Vector_Size = rho
    #         self.target_X = rho * math.cos(phi)
    #         self.target_Y = rho * math.sin(phi)
    #         if self.Target_Vector_Orientation > math.pi * 2:
    #             self.Target_Vector_Orientation = self.Target_Vector_Orientation - (math.pi * 2)
    #         if self.Target_Vector_Orientation < 0:
    #             self.Target_Vector_Orientation = (math.pi * 2) + self.Target_Vector_Orientation
    #
    #         self.train_data[i][0] = self.Target_Vector_Size
    #         self.train_data[i][1] = self.Target_Vector_Orientation
    #     test_in_line = 0

    # Load Visualization files
    def load_data(self, slot_number, train=True):
        if train:
            print('Loading Training data from reservoir has started')
            self.actions_vis = pickle.load(open(self.media_address + 'Reservoir/Train/Action' + str(slot_number) +
                                                '.p', "rb"))
            self.states_vis = pickle.load(open(self.media_address + 'Reservoir/Train/State' + str(slot_number) +
                                               '.p', "rb"))

    def load_model(self):
        self.actor.model.load_weights("actor_modelStaticSize0.h5")
        self.critic.model.load_weights("critic_modelStaticSize0.h5")
        self.actor.target_model.load_weights("actor_modelStaticSize0.h5")
        self.critic.target_model.load_weights("critic_modelStaticSize0.h5")

    def main(self, v_rep=False):
        total_iteration_num = 0
        for i in range(1, self.Episode_N, 1):
            total_reward = 0
            self.env.init()
            observation, hindsight_s, hindsight_s1, reward, done = self.env.step((200, 200), True)
            s_t = np.array(observation).reshape(7)
            for j in range(self.steps_max):  # Iteration Number
                total_iteration_num = total_iteration_num + 1
                loss = 0
                self.hindsight = False
                if np.random.random() > self.EPSILON:
                    if np.random.random() < self.EPSILON:
                        self.hindsight = True
                    self.hindsight = True
                    a_type = "Exploit"
                    a_t = self.actor.model.predict(s_t.reshape(1, s_t.shape[0]))[0]  # rescale
                    a_t_x = self.env.min_pos[0] + ((self.env.max_pos[0] - self.env.min_pos[0]) * a_t[0])
                    a_t_y = self.env.min_pos[1] + ((self.env.max_pos[1] - self.env.min_pos[1]) * a_t[1])
                    a_t = np.array((a_t_x, a_t_y))
                else:
                    if np.random.random() > self.EPSILON:
                        self.hindsight = True
                    a_type = "Explore"
                    a_t_x = np.random.uniform(-self.env.min_pos[0], self.env.max_pos[0], size=(1, 1))
                    a_t_y = np.random.uniform(-self.env.min_pos[1], self.env.max_pos[1], size=(1, 1))
                    a_t = np.array((a_t_x[0, 0], a_t_y[0, 0]))

                #  running simulator for the step
                observation, hindsight_s,hindsight_s1, r_t, done = self.env.step(a_t, False)

                #  storing next state
                s_t1 = np.array(observation).reshape(7)
                a_t[0] = a_t[0] / 600
                a_t[1] = a_t[1] / 400
                if self.hindsight:
                    # Save to memory buffer
                    for p in range(len(hindsight_s)):
                        self.buffer.add_main_memory(np.array(hindsight_s[p]).reshape(7), a_t, 1, np.array(hindsight_s1[p]).reshape(7), False)


                self.buffer.add_main_memory(np.float16(s_t), np.float16(a_t), np.float16(r_t), np.float16(s_t1),done)
                batch = self.buffer.get_batch_main_memory(self.BATCH_SIZE )
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[2] for e in batch])
                #  getting Q value from critic network given actor network
                target_q_values = self.critic.target_model.predict(
                    [new_states, self.actor.target_model.predict(new_states)])  # new_states->actions actor output

                #  discounting reward manipulation
                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + self.ALPHA * target_q_values[k]

                if self.TRAIN:
                    #  getting critic loss
                    loss += self.critic.model.train_on_batch([states, actions], y_t)

                    #  getting next action based on loaded states from memory
                    a_for_grad = self.actor.model.predict(states)

                    #  update critic gradients by next action
                    grads = self.critic.gradients(states, a_for_grad)

                    #  train actor network
                    self.actor.train(states, grads)
                    #  train actor target network
                    self.actor.target_train()
                    #  train critic target network
                    self.critic.target_train()
                total_reward += r_t
                s_t = s_t1
                if self.TERMINAL_WRITE:
                    print("-------------------")
                    print("Action", a_type)
                    print("Episode", i, "Step", i, "Reward", r_t, "Loss", loss, "Epsilon", self.EPSILON)
                    print("X1 Y1" + str(a_t[0]) + "    " + str(a_t[1]))
                    print("total reward" + str(total_reward))
                    print("-------------------")

                # output_str = output_str + "\t" + str(s) + "\t" + str(i) + "\t" + str(j) + "\t" + str(self.step) \
                #              + "\t" + str(a_type) + "\t" + str(
                #     r_t) + "\t" + str(loss) + "\t" + str(self.EPSILON) + "\t" + str(a_t[0][0]) + "\t" + str(
                #     a_t[0][1]) + "\t" + str(a_t[0][1]) + "\t" + str(total_reward) + "\t" + str(self.target_size) \
                #              + "\t" + str(self.Target_Vector_Size) + "\t" + str(self.Target_Vector_Orientation) + "\n"
                if done:
                    self.step += 1
                    if self.success_queue.qsize() > self.check_queue_size - 1:
                        self.success_queue.get()
                    self.success_queue.put(1)
                    break
                if j == self.steps_max - 1:
                    if self.success_queue.qsize() > self.check_queue_size - 1:
                        self.success_queue.get()
                    self.success_queue.put(0)

            print("Now we save model")

            # if self.DYNAMIC_SIZE_TARGET:
            #     self.actor.model.save_weights("actor_modelDynamicSize" + str(self.run_Num) + self.define + ".h5",
            #                                   overwrite=True)
            #     self.critic.model.save_weights("critic_modelDynamicSize" + str(self.run_Num) + self.define + ".h5",
            #                                    overwrite=True)
            #     pickle.dump(output_str, open(
            #         "DynamicSize_Results" + str(self.run_Num) + 'S' + str(self.Current_Story) + self.define +
            #         '.p', "wb"))
            # else:
            #     self.actor.model.save_weights("actor_modelStaticSize" + str(self.run_Num) + self.define + ".h5",
            #                                   overwrite=True)
            #     self.critic.model.save_weights("critic_modelStaticSize" + str(self.run_Num) + self.define + ".h5",
            #                                    overwrite=True)
            #     pickle.dump(output_str, open(
            #         "StaticSize_Results" + str(self.run_Num) + 'S' + str(self.Current_Story) +
            #         self.define + '.p', "wb"))

            output_str = ""
        # self.actual_run(

    def gpu_setup(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        K.set_session(self.sess)


    # Generates the plot (output of critic and actor) for the current state_vis
    def plot_generator(self):
        actor_prediction_part_1 = self.actor.target_model.predict_on_batch([self.states_vis[0]])
        actor_prediction_part_2 = self.actor.target_model.predict_on_batch([self.states_vis[1]])
        actor_prediction_part_3 = self.actor.target_model.predict_on_batch([self.states_vis[2]])
        actor_prediction_part_4 = self.actor.target_model.predict_on_batch([self.states_vis[3]])
        actor_prediction_part_5 = self.actor.target_model.predict_on_batch([self.states_vis[4]])
        actor_prediction_part_6 = self.actor.target_model.predict_on_batch([self.states_vis[5]])
        actor_prediction_part_7 = self.actor.target_model.predict_on_batch([self.states_vis[6]])
        actor_prediction_part_8 = self.actor.target_model.predict_on_batch([self.states_vis[7]])
        actor_prediction_part_9 = self.actor.target_model.predict_on_batch([self.states_vis[8]])
        g1_first_joint = np.array(actor_prediction_part_1[:, 0])
        g2_first_joint = np.array(actor_prediction_part_2[:, 0])
        g3_first_joint = np.array(actor_prediction_part_3[:, 0])
        g4_first_joint = np.array(actor_prediction_part_4[:, 0])
        g5_first_joint = np.array(actor_prediction_part_5[:, 0])
        g6_first_joint = np.array(actor_prediction_part_6[:, 0])
        g7_first_joint = np.array(actor_prediction_part_7[:, 0])
        g8_first_joint = np.array(actor_prediction_part_8[:, 0])
        g9_first_joint = np.array(actor_prediction_part_9[:, 0])
        g1_second_joint = np.array(actor_prediction_part_1[:, 1])
        g2_second_joint = np.array(actor_prediction_part_2[:, 1])
        g3_second_joint = np.array(actor_prediction_part_3[:, 1])
        g4_second_joint = np.array(actor_prediction_part_4[:, 1])
        g5_second_joint = np.array(actor_prediction_part_5[:, 1])
        g6_second_joint = np.array(actor_prediction_part_6[:, 1])
        g7_second_joint = np.array(actor_prediction_part_7[:, 1])
        g8_second_joint = np.array(actor_prediction_part_8[:, 1])
        g9_second_joint = np.array(actor_prediction_part_9[:, 1])
        g1_first_mean = np.mean(g1_first_joint.reshape(-1, 9), axis=1)
        g2_first_mean = np.mean(g2_first_joint.reshape(-1, 9), axis=1)
        g3_first_mean = np.mean(g3_first_joint.reshape(-1, 9), axis=1)
        g4_first_mean = np.mean(g4_first_joint.reshape(-1, 9), axis=1)
        g5_first_mean = np.mean(g5_first_joint.reshape(-1, 9), axis=1)
        g6_first_mean = np.mean(g6_first_joint.reshape(-1, 9), axis=1)
        g7_first_mean = np.mean(g7_first_joint.reshape(-1, 9), axis=1)
        g8_first_mean = np.mean(g8_first_joint.reshape(-1, 9), axis=1)
        g9_first_mean = np.mean(g9_first_joint.reshape(-1, 9), axis=1)
        g1_second_mean = np.mean(g1_second_joint.reshape(-1, 9), axis=1)
        g2_second_mean = np.mean(g2_second_joint.reshape(-1, 9), axis=1)
        g3_second_mean = np.mean(g3_second_joint.reshape(-1, 9), axis=1)
        g4_second_mean = np.mean(g4_second_joint.reshape(-1, 9), axis=1)
        g5_second_mean = np.mean(g5_second_joint.reshape(-1, 9), axis=1)
        g6_second_mean = np.mean(g6_second_joint.reshape(-1, 9), axis=1)
        g7_second_mean = np.mean(g7_second_joint.reshape(-1, 9), axis=1)
        g8_second_mean = np.mean(g8_second_joint.reshape(-1, 9), axis=1)
        g9_second_mean = np.mean(g9_second_joint.reshape(-1, 9), axis=1)
        g_first_total = np.append(g1_first_mean, g2_first_mean, axis=0)
        g_first_total = np.append(g_first_total, g3_first_mean, axis=0)
        g_first_total = np.append(g_first_total, g4_first_mean, axis=0)
        g_first_total = np.append(g_first_total, g5_first_mean, axis=0)
        g_first_total = np.append(g_first_total, g6_first_mean, axis=0)
        g_first_total = np.append(g_first_total, g7_first_mean, axis=0)
        g_first_total = np.append(g_first_total, g8_first_mean, axis=0)
        g_first_total = np.append(g_first_total, g9_first_mean, axis=0)
        g_second_total = np.append(g1_second_mean, g2_second_mean, axis=0)
        g_second_total = np.append(g_second_total, g3_second_mean, axis=0)
        g_second_total = np.append(g_second_total, g4_second_mean, axis=0)
        g_second_total = np.append(g_second_total, g5_second_mean, axis=0)
        g_second_total = np.append(g_second_total, g6_second_mean, axis=0)
        g_second_total = np.append(g_second_total, g7_second_mean, axis=0)
        g_second_total = np.append(g_second_total, g8_second_mean, axis=0)
        g_second_total = np.append(g_second_total, g9_second_mean, axis=0)
        plot_first_joint = g_first_total.reshape(-1, 360)
        plot_second_joint = g_second_total.reshape(-1, 360)
        g1 = self.critic.target_model.predict_on_batch([self.states_vis[0], self.actions_vis[0]])
        g2 = self.critic.target_model.predict_on_batch([self.states_vis[1], self.actions_vis[1]])
        g3 = self.critic.target_model.predict_on_batch([self.states_vis[2], self.actions_vis[2]])
        g4 = self.critic.target_model.predict_on_batch([self.states_vis[3], self.actions_vis[3]])
        g5 = self.critic.target_model.predict_on_batch([self.states_vis[4], self.actions_vis[4]])
        g6 = self.critic.target_model.predict_on_batch([self.states_vis[5], self.actions_vis[5]])
        g7 = self.critic.target_model.predict_on_batch([self.states_vis[6], self.actions_vis[6]])
        g8 = self.critic.target_model.predict_on_batch([self.states_vis[7], self.actions_vis[7]])
        g9 = self.critic.target_model.predict_on_batch([self.states_vis[8], self.actions_vis[8]])
        g1_mean = np.mean(g1.reshape(-1, 9), axis=1)
        g2_mean = np.mean(g2.reshape(-1, 9), axis=1)
        g3_mean = np.mean(g3.reshape(-1, 9), axis=1)
        g4_mean = np.mean(g4.reshape(-1, 9), axis=1)
        g5_mean = np.mean(g5.reshape(-1, 9), axis=1)
        g6_mean = np.mean(g6.reshape(-1, 9), axis=1)
        g7_mean = np.mean(g7.reshape(-1, 9), axis=1)
        g8_mean = np.mean(g8.reshape(-1, 9), axis=1)
        g9_mean = np.mean(g9.reshape(-1, 9), axis=1)
        g_total = np.append(g1_mean, g2_mean, axis=0)
        g_total = np.append(g_total, g3_mean, axis=0)
        g_total = np.append(g_total, g4_mean, axis=0)
        g_total = np.append(g_total, g5_mean, axis=0)
        g_total = np.append(g_total, g6_mean, axis=0)
        g_total = np.append(g_total, g7_mean, axis=0)
        g_total = np.append(g_total, g8_mean, axis=0)
        g_total = np.append(g_total, g9_mean, axis=0)
        plot = g_total.reshape(-1, 360)
        return plot, plot_first_joint, plot_second_joint

    # Main Visualizing Function which visualizes the target
    def map_visualizer(self, episod_num, is_story):
        self.target_size = self.TARGET_MINIMUM_SIZE
        if not is_story:
            joint_space = self.env.joint_space_visualizer(self.Target_Vector_Orientation, self.Target_Vector_Size,
                                                          self.target_size)
            plot, plot_first_joint, plot_second_joint = self.plot_generator()

            pickle.dump(plot_first_joint, open(self.media_address + self.Address + 'Training/ActorMapJoint0/S' +
                                               str(self.Current_Story) + 'E'
                                               + str(episod_num) + 'ti' + str(self.target_index) + self.define + '.p',
                                               "wb"))
            pickle.dump(plot_second_joint, open(self.media_address + self.Address + 'Training/ActorMapJoint1/S' +
                                                str(self.Current_Story) + 'E'
                                                + str(episod_num) + 'ti' + str(self.target_index) + self.define + '.p',
                                                "wb"))
            pickle.dump(plot, open(self.media_address + self.Address + 'Training/CriticMap/S' +
                                   str(self.Current_Story) + 'E' + str(episod_num)
                                   + 'ti' + str(self.target_index) + self.define + '.p', "wb"))
            pickle.dump(joint_space, open(self.media_address + self.Address + 'Training/JointSpace/S' +
                                          str(self.Current_Story) + 'E' + str(episod_num)
                                          + 'ti' + str(self.target_index) + self.define + '.p', "wb"))
        else:
            print('Now we are working on visualization data with Trajectory')
            counter_success_train = 0
            success = np.zeros(self.env.tracker.target_position.shape[0])
            rand_max = self.env.tracker.target_position.shape[0]
            success_index = ""
            for i in range(0, rand_max, 1):
                done = self.evaluate_run(True, i)
                if done:
                    counter_success_train += 1
                    success[i] = 1
                    success_index += "1 "
                else:
                    success_index += "0 "
                if self.Visualizer:
                    self.load_data(i, True)
                    joint_space = self.env.joint_space_visualizer(self.Target_Vector_Orientation,
                                                                  self.Target_Vector_Size, self.target_size)

                    plot, plot_first_joint, plot_second_joint = self.plot_generator()
                    pickle.dump(plot_first_joint, open(self.media_address + self.Address +
                                                       'Trajectory/ActorMapJoint0/S-' +
                                                       str(self.Current_Story) + 'tri' + str(
                        self.target_index) + self.define + '.p',
                                                       "wb"))
                    pickle.dump(plot_second_joint, open(self.media_address + self.Address +
                                                        'Trajectory/ActorMapJoint1/S-' +
                                                        str(self.Current_Story) + 'tri' +
                                                        str(self.target_index) + self.define + '.p', "wb"))
                    pickle.dump(plot, open(
                        self.media_address + self.Address + 'Trajectory/CriticMap/S-'
                        + str(self.Current_Story) + 'tri' +
                        str(self.target_index) + self.define + '.p', "wb"))
                    pickle.dump(joint_space, open(self.media_address + self.Address + 'Trajectory/JointSpace/S-' +
                                                  str(self.Current_Story) + 'tri'
                                                  + str(self.target_index) + self.define + '.p', "wb"))
                    pickle.dump(self.angle_buffer, open(
                        self.media_address + self.Address
                        + 'Trajectory/Trajectory/S-' + str(self.Current_Story) + 'tri' +
                        str(self.target_index) + self.define + '.p', "wb"))
            counter_success_test = 0
            rand_max = len(self.test_data)
            while True:
                try:
                    if self.DYNAMIC_SIZE_TARGET:
                        result_file = open("DynamicSize_sucess" + str(self.run_Num) + self.define + ".txt", 'a')
                    else:
                        result_file = open("StaticSize_sucess" + str(self.run_Num) + self.define + ".txt", 'a')
                    result_str = "Story " + str(self.Current_Story) + " " + str(counter_success_train) + " " + str(
                        counter_success_test) + " " + success_index + "\n"
                    print result_str
                    result_file.write(result_str)
                    result_file.close()
                    break
                except:
                    self.email.send_email("whatsuploop@gmail.com", "RickandMorthy", "hadi.beikmohammadi@gmail.com",
                                          "Exception " + self.define,
                                          "Story")

                    time.sleep(300)

    # Generate Visualization data it should be called just once
    def generate_visualization_data(self, radius, orientation):
        print('generating data for visualization ...')
        if self.Visualizer:
            print('Process Visualizer File')
            for z in range(0, 360, 1):
                for u in range(0, 360, 1):
                    for r in range(-1, 2, 1):
                        for t in range(-1, 2, 1):
                            g = [math.sin(math.radians(z)), math.cos(math.radians(z)), math.sin(math.radians(u)),
                                 math.cos(math.radians(u)), radius / 200, math.sin(orientation), math.cos(orientation)]
                            new_state = np.asarray(g)
                            e = [r, t]
                            new_action = np.asarray(e)
                            self.vis_buffer.add_visualization(new_state, new_action)
            batch = self.vis_buffer.getbatch_vis(360 * 360 * 9)
            self.states_vis = np.split(np.asarray([e[0] for e in batch]), 9)
            self.actions_vis = np.split(np.asarray([e[1] for e in batch]), 9)


if __name__ == "__main__":
    # print(str(sys.argv[0]))
    # print(str(sys.argv[1]))
    # print(str(sys.argv[2]))
    # ddpg_Agent = DdpgAgent(int(sys.argv[1]), int(sys.argv[2]))

    ddpg_Agent = DdpgAgent(0, 0)
    ddpg_Agent.main()
