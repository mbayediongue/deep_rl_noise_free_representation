import numpy as np
import abc
import cv2 as cv
from scipy.spatial.distance import cdist

class AbstractReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def add(self, time_step):
        pass

    @abc.abstractmethod
    def __next__(self, ):
        pass

    @abc.abstractmethod
    def __len__(self, ):
        pass


class EfficientReplayBufferContrastive(AbstractReplayBuffer):
    '''Fast + efficient replay buffer implementation in numpy.'''

    def __init__(self, buffer_size, batch_size, nstep, discount, frame_stack,
                 data_specs=None, sarsa=False, constrastive_sub_buffer_size=None):
        self.buffer_size = buffer_size
        self.data_dict = {}
        self.index = -1
        self.traj_index = 0
        self.frame_stack = frame_stack
        self._recorded_frames = frame_stack + 1
        self.batch_size = batch_size
        self.nstep = nstep
        self.discount = discount
        self.full = False
        self.discount_vec = np.power(discount, np.arange(nstep))  # n_step - first dim should broadcast
        self.next_dis = discount ** nstep
        self.sarsa = sarsa
        if constrastive_sub_buffer_size is None:
            self.constrastive_sub_buffer_size = batch_size
    

    def _initial_setup(self, time_step):
        self.index = 0
        self.obs_shape = list(time_step.observation.shape)
        self.ims_channels = self.obs_shape[0] // self.frame_stack
        self.act_shape = time_step.action.shape

        self.obs = np.zeros([self.buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8)
        self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)
        self.k_step = np.ones([self.buffer_size], dtype=np.float32)
    
    def _initial_sub_contrastive_buffer(self, constrastive_sub_buffer_size):
        self.indices_constrast_buffer = np.random.choice(self.index, constrastive_sub_buffer_size)
        #self.sub_buffer_obs = np.zeros([self.contrastive_buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8)
        #self.hsv_ims = np.zeros([constrastive_sub_buffer_size, self.h_bins, self.s_bins], dtype=np.uint8)
        #indices_constrast_buffer = np.random.choice(index, contrastive_sub_buffer_size, replace=False)
        #im = np.transpose(episodes['observation'][-1], (1,2, 0))

    def rgb_to_hsv_buffer(self):
        #histSize = [self.h_bins, self.s_bins]
        #ranges = self.h_ranges + self.s_ranges # concat lists
        # Use the 0-th and 1-st channels
        #channels = [0, 1]
        for i in range(len(self.indices_constrast_buffer)):
            idx = self.indices_constrast_buffer[i]
            self.hsv_ims[i] = cv.cvtColor(np.transpose(self.obs[idx], (1,2,0)), cv.COLOR_RGB2HSV)
            #hist_im = cv.calcHist([hsv_im], channels, None, histSize, ranges, accumulate=False)
            #cv.normalize(hist_im, hist_im, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            #self.histograms[i] = hist_im
    
    def update_sub_buffer(self, nb_indices_to_update=None):
        if nb_indices_to_update is None:
            nb_indices_to_update = self.constrastive_sub_buffer_size//5
        if nb_indices_to_update > self.constrastive_sub_buffer_size:
            nb_indices_to_update = self.constrastive_sub_buffer_size
            print(f'The number of buffer elements [{nb_indices_to_update}] to update is greater to the sub buffer'+
            ' size ===> updating all the sub buffer ... \n')
    
        updated_indices = np.random.choice(self.constrastive_sub_buffer_size, size=nb_indices_to_update)
        self.indices_constrast_buffer[updated_indices] = np.random.choice(self.index, nb_indices_to_update)

        for i in range(len(updated_indices)):
            idx = self.histograms[updated_indices[i]]
            hsv_im = cv.cvtColor(np.transpose(self.obs[idx], (1,2,0)), cv.COLOR_RGB2HSV)
            self.hsv_ims[updated_indices[i]] = hsv_im

    def rgb_to_hsv(self, observations):
        hsv_ims = np.zeros([len(observations), self.h_bins, self.s_bins], dtype=np.float32)
        for i in range(len(observations)):
            hsv_im = cv.cvtColor(np.transpose(observations[i], (1,2,0)), cv.COLOR_RGB2HSV)
            hsv_ims[i] = hsv_im
        return hsv_ims
    

    def update_histograms_sub_buffer(self, nb_indices_to_update=None):
        if nb_indices_to_update is None:
            nb_indices_to_update = self.constrastive_sub_buffer_size//5
        updated_indices = np.random.choice(self.constrastive_sub_buffer_size, size=nb_indices_to_update)
        self.indices_constrast_buffer[updated_indices] = np.random.choice(self.index, nb_indices_to_update)

        histSize = [self.h_bins, self.s_bins]
        ranges = self.h_ranges + self.s_ranges # concat lists
        # Use the 0-th and 1-st channels
        channels = [0, 1]
        for i in range(len(updated_indices)):
            idx = self.histograms[updated_indices[i]]
            hsv_im = cv.cvtColor(np.transpose(self.obs[idx], (1,2,0)), cv.COLOR_BGR2HSV)
            hist_im = cv.calcHist([hsv_im], channels, None, histSize, ranges, accumulate=False)
            cv.normalize(hist_im, hist_im, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            self.histograms[updated_indices[i]] = hist_im

    def get_constrastive_states(self, indices, weight_actions=1.):
        """
        For each pair (observation, action) in the (self.obs[indices], self.act[indices]), 
        we look for the the corresponding onstrastive pair (const_obs, const_action) in the
        constrastive_sub_buffer which minimize a criterion such that:
            - observation and const_obs are SIMILAR as possible
            - action and dissimilarities are DISSIMILAR as possible

        The criterion to minimize is built by using a simple linear combination:
            criterion =  observation_similarity - weight_actions * action_similarity

        weigth_actions : weights of actions dissimilarities in the criterion
        """
 
        base_obs = self.obs[indices]

        actions_distances = cdist(self.act[indices], self.act[self.indices_constrast_buffer]).reshape((-1))
        sub_buff_sz = self.indices_constrast_buffer
        states_distances = cdist(base_obs.reshape((len(base_obs), -1)),
                self.obs[self.indices_constrast_buffer].reshape(sub_buff_sz, -1)).reshape((-1))
        # normalization
        states_distances = (states_distances-states_distances.mean())/np.std(states_distances)
        actions_distances = (actions_distances-actions_distances.mean())/np.std(actions_distances)

        # criterion: similarity between observations and dissimilarity between actions
        criterions = states_distances - weight_actions *  actions_distances 

        return np.argmin(criterions, axis=1)
 

    def add_data_point(self, time_step):
        first = time_step.first()
        latest_obs = time_step.observation[-self.ims_channels:]
        if first:
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index:self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.full = True
                else:
                    self.obs[self.index:end_index] = latest_obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index:self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                self.obs[self.index:end_index] = latest_obs
                self.valid[self.index:end_invalid] = False
            self.index = end_index
            self.traj_index = 1
        else:
            np.copyto(self.obs[self.index], latest_obs)  # Check most recent image
            np.copyto(self.act[self.index], time_step.action)
            self.rew[self.index] = time_step.reward
            self.dis[self.index] = time_step.discount
            self.valid[(self.index + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.index - self.nstep + 1) % self.buffer_size] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def add(self, time_step):
        if self.index == -1:
            self._initial_setup(time_step)
        self.add_data_point(time_step)

    def __next__(self, ):
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.gather_nstep_indices(indices)

    def gather_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        #gather_ranges = all_gather_ranges[:, self.frame_stack:]  # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        #nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]

        #obs = self.obs[obs_gather_ranges]
        
        if self.index >= self.constrastive_sub_buffer_size :
            # change all the constrastive sub buffer elements
            self.indices_constrast_buffer(self.constrastive_sub_buffer_size)
            indices2 = self.get_similar_states_indices(obs_gather_ranges[:,0])
        else:
            indices2 = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)

        #obs = np.reshape(obs, [n_samples, *self.obs_shape])
        #nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape])

        #act = self.act[indices]
        #dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        k_step1 = self.k_step[indices].astype(int)
        k_step_rand1 = []
        for each in k_step1:
            if each > 1:
                k_step_rand1.append(np.random.randint(low=1, high=each))
            else:
                k_step_rand1.append(1)
        k_all_gather_ranges = np.stack([np.arange(indices[i] + k_step_rand1[i] - self.frame_stack, indices[i] + k_step_rand1[i] + self.nstep)
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        k_obs_gather_ranges = k_all_gather_ranges[:, :self.frame_stack]
        k_next_obs_gather_ranges = k_all_gather_ranges[:, -self.frame_stack:]
        obs_k_1 = np.reshape(self.obs[k_obs_gather_ranges], [n_samples, *self.obs_shape])

        gather_ranges = k_all_gather_ranges[:, self.frame_stack:]  # bs x nstep
        all_rewards = self.rew[gather_ranges]
        # Could implement below operation as a matmul in pytorch for marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)
        nobs = np.reshape(self.obs[k_next_obs_gather_ranges], [n_samples, *self.obs_shape])
        act = self.act[indices+k_step_rand1]
        dis = np.expand_dims(self.next_dis * self.dis[k_next_obs_gather_ranges[:, -1]], axis=-1)

        # Get the k-step constrastive observation associated with each k observation
        k_step2 = self.k_step[indices2].astype(int)
        k_step_rand2 = []
        for each in k_step2:
            if each > 1:
                k_step_rand2.append(np.random.randint(low=1, high=each))
            else:
                k_step_rand2.append(1)
        k_all_gather_ranges = np.stack([np.arange(indices2[i] + k_step_rand2[i] - self.frame_stack, indices2[i] + k_step_rand2[i] + self.nstep)
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        k_obs_gather_ranges = k_all_gather_ranges[:, :self.frame_stack]
        obs_k_2 = np.reshape(self.obs[k_obs_gather_ranges], [n_samples, *self.obs_shape])
        act2 = self.act[indices2+k_step_rand2]

        if self.sarsa:
            nact = self.act[indices+k_step_rand1+ self.nstep]
            return (obs_k_1, act, rew, dis, nobs, nact)

        return (obs_k_1, act, rew, dis, nobs, obs_k_2, act2)

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index

    def get_train_and_val_indices(self, validation_percentage):
        all_indices = self.valid.nonzero()[0]
        num_indices = all_indices.shape[0]
        num_val = int(num_indices * validation_percentage)
        np.random.shuffle(all_indices)
        val_indices, train_indices = np.split(all_indices,
                                              [num_val])
        return train_indices, val_indices

    def get_obs_act_batch(self, indices):
        n_samples = indices.shape[0]
        obs_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i])
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        act = self.act[indices]
        return obs, act
    

