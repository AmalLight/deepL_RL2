#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Modified by Jeremi Kaczmarczyk (jeremi.kaczmarczyk@gmail.com) 2018 
# For Udacity Deep Reinforcement Learning Nanodegree

import numpy as np
import torch
import torch.nn as nn

from imported_utils import Batcher


class PPOAgent(object):
    
    def __init__(self, environment, brain_name, policy_network, optimizier, config):
        self.config = config
        self.hyperparameters = config['hyperparameters']
        self.network = policy_network
        self.optimizier = optimizier
        self.total_steps = 0
        self.all_rewards = np.zeros(config['environment']['number_of_agents'])
        self.episode_rewards = []
        self.environment = environment
        self.brain_name = brain_name
        
        env_info = environment.reset(train_mode=True)[brain_name]    
        self.states = env_info.vector_observations              

    def step(self):
        rollout = []
        hyperparameters = self.hyperparameters

        env_info = self.environment.reset(train_mode=True)[self.brain_name]    
        self.states = env_info.vector_observations  
        states = self.states

        # print ( self.config['environment']['number_of_agents'] ) # 20

        step_reached = 0
        for l in range(hyperparameters['rollout_length']):
            actions, log_probs, _, values = self.network(states)
            env_info = self.environment.step(actions.cpu().detach().numpy())[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            terminals = np.array([1 if t else 0 for t in env_info.local_done])
            self.all_rewards += rewards

            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.all_rewards[i])
                    self.all_rewards[i] = 0

            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals])
            states = next_states

            step_reached = l+1
            # print ( len(rollout) ) # 1,2,3,4,..

        # print ( step_reached ) # 2048
        # print ( len(rollout) ) # 2048

        self.states = states
        pending_value = self.network(states)[-1]
        rollout.append([states, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = torch.Tensor(np.zeros((self.config['environment']['number_of_agents'], 1)))
        returns = pending_value.detach()

        step_reached = 0
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = torch.Tensor(terminals).unsqueeze(1)
            rewards = torch.Tensor(rewards).unsqueeze(1)
            actions = torch.Tensor(actions)
            states = torch.Tensor(states)
            next_value = rollout[i + 1][1]

            # print ( 'next_value:' , next_value.shape ) # 20,1
            # if i == ( len ( rollout ) - 2 ) :
            #    print ( 'next_value:'    ,    next_value ) # equals
            #    print ( 'pending_value:' , pending_value ) # equals
            #    print ( 'returns:'       , returns       ) # equals

            returns  = rewards + hyperparameters['discount_rate'] * terminals * returns
            td_error = rewards + hyperparameters['discount_rate'] * terminals * next_value.detach () - \
                                                                                     value.detach ()

            advantages = advantages * hyperparameters['tau'] * hyperparameters['discount_rate'] * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

            step_reached = i+1

        # print ( step_reached ) # 1 <= for reversed

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        # print ( states.shape ) # 40960 => flatten

        batcher = Batcher(states.size(0) // hyperparameters['mini_batch_number'], [np.arange(states.size(0))])

        groups_count = 0
        for _ in range(hyperparameters['optimization_epochs']):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                groups_count += 1

                # print ( 'batch:' , batch_indices.shape ) # 1280,1

                batch_indices = torch.Tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                # print ( 'states:' , sampled_states.shape ) # 1280,33

                _, log_probs, entropy_loss, values = self.network(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - hyperparameters['ppo_clip'],
                                          1.0 + hyperparameters['ppo_clip']) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - hyperparameters['entropy_coefficent'] * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.optimizier.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), hyperparameters['gradient_clip'])
                self.optimizier.step()

            # print ( groups_count ) # 32

        steps = hyperparameters['rollout_length'] * self.config['environment']['number_of_agents']
        self.total_steps += steps
