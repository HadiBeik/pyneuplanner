from collections import deque
import random
import sys
import pickle


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size

        self.num_experiences = 0
        self.num_experiences1 = 0

        self.buffer = deque()
        self.buffer1 = deque()

        self.local_memory = deque()
        self.main_memory = deque()
        self.num_experience_local_memory = 0
        self.num_experience_main_memory = 0

    #  Main memory to use for Memory modification for filtered data
    def add_main_memory(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experience_main_memory < self.buffer_size:
            self.main_memory.append(experience)
            self.num_experience_main_memory += 1
        else:
            self.main_memory.popleft()
            self.main_memory.append(experience)
        # print("Buffer length M: " + str(self.num_experience_main_memory) + " Buffer size: " + str(sys.getsizeof(self.main_memory)))

    def get_batch_main_memory(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experience_main_memory < batch_size:
            return random.sample(self.main_memory, self.num_experience_main_memory)
        else:
            return random.sample(self.main_memory, batch_size)

    def save_physical(self, dynamic=True):
        media_address = "/media/hadi/646C75BE6C758C14/DDPG-Repo/Data/Buffer/"
        if dynamic:
            pickle.dump(self.main_memory, open(media_address + "DynamicBuffer" + ".p", "wb"))
        else:
            pickle.dump(self.main_memory, open(media_address + "StaticBuffer" + ".p", "wb"))

    def load_physical(self, dynamic=True):
        media_address = "/media/hadi/646C75BE6C758C14/DDPG-Repo/Data/Buffer/"
        if dynamic:
            self.main_memory = pickle.load(open(media_address + "DynamicBuffer" + ".p", "rb"))
        else:
            self.main_memory = pickle.load(open(media_address + "StaticBuffer" + ".p", "rb"))

    def reset_local_memory(self):
        self.local_memory = self.main_memory
        self.num_experience_local_memory = self.num_experience_main_memory

    ########################################
    ########     Visualization     #########
    ########################################

    # Build state action visualization data exactly like input of network
    def add_visualization(self, state, action):
        experience = (state, action)
        if self.num_experiences1 < self.buffer_size:
            self.buffer1.append(experience)
            self.num_experiences1 += 1
        else:
            self.buffer1.popleft()
            self.buffer1.append(experience)

    #  get batch for visualizing data
    def getbatch_vis(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences1 < batch_size:
            return self.buffer1
        else:
            return self.buffer1

    ########################################
    #####     Memory Modification     ######
    ########################################

    #  General add function in the experiments which we don't have memory modification

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
