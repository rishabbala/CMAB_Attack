import random
import numpy as np
from scipy.optimize import minimize
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import random



class CUCB():

    def __init__(self, args, arms_set=None, mu=None, attack=False):

        self.args = args  

        # self.theta_star = []
        # for _ in range(args.feature_dimension):
        #     self.theta_star.append(random.uniform(0, 2))
        # self.theta_star = np.array(self.theta_star)
        # self.theta_star /= np.sqrt(np.sum(np.square(self.theta_star)))
        # self.theta_star *= random.uniform(0.3, 1)
        # self.reward_set_theta_star = []

        self.attack = attack
        if arms_set == None:
            self.arms_set = range(0, self.args.num_arms)
            self.mu = []
            for i in range(self.args.num_arms):
                self.mu.append(np.random.uniform(0.1, 0.9, 1)[0])
        
        else:
            self.arms_set = arms_set
            self.mu = mu

        self.optimal_arm_reward = sum(sorted(self.mu, reverse=True)[:self.args.K])
        self.target_arms = random.sample(range(self.args.num_arms), self.args.K)

        # for i in range(self.args.num_arms):
        #     arm = []
        #     for _ in range(self.args.feature_dimension):
        #         arm.append(random.uniform(0, 2))
        #     arm = np.array(arm)
        #     arm /= np.sqrt(np.sum(np.square(arm)))
        #     arm *= random.uniform(0.3, 1)
            
        #     if type(self.arms_set).__name__ == 'list':
        #         self.arms_set = arm.reshape(1, self.args.feature_dimension)
        #     else:
        #         self.arms_set = np.append(self.arms_set, [arm], axis=0)

        #     self.reward_set_theta_star.append([np.dot(arm, self.theta_star), i])

        # self.reward_set_theta_star = sorted(self.reward_set_theta_star, reverse=True)

        # self.optimal_arms = self.reward_set_theta_star[:self.args.K]
        # self.optimal_arm_reward = 0
        # for i in self.optimal_arms:
        #     self.optimal_arm_reward += i[0]

        self.cumulative_regret = []
        self.all_reward = []
        self.cumulative_cost = []

        self.thresh = 0.01
        self.target_reward_min = 1
        self.R = 0.01
        self.t = 0
        self.T = [0]*self.args.num_arms
        self.mu_hat = [1]*self.args.num_arms
        # for i in range(self.args.num_arms-1):
        #     self.mu_hat = np.concatenate((self.mu_hat, np.array([1]*self.args.feature_dimension).reshape(1, -1)))


    def step(self):

        self.t += 1
        self.reward = 0
        self.regret = 0
        self.cost = 0

        self.mu_bar = []
        for i in range(self.args.num_arms): 
            if self.T[i] == 0:
                self.mu_bar.append([1, i])
            else:
                self.mu_bar.append([min(self.mu_hat[i] + np.sqrt(3*np.log(self.t)/(2*self.T[i])), 1), i])

        solution_set = self.oracle()

        if not self.attack:
            recieved_reward = []
            for i in solution_set:
                r = min(max(self.mu[i[1]] + np.random.normal(0, self.R, 1)[0], 0), 1)
                recieved_reward.append(r)
                self.reward += r
            self.cost = 0            
        
        else:
            recieved_reward = self.model_attack(solution_set)

        self.regret = self.optimal_arm_reward - self.reward

        if self.t == 1:
            self.cumulative_regret.append(self.regret)
            self.cumulative_cost.append(self.cost)
        else:
            self.cumulative_regret.append(self.cumulative_regret[-1] + self.regret)
            self.cumulative_cost.append(self.cumulative_cost[-1] + self.cost)

        self.all_reward.append(self.reward)

        if self.cost < 0:
            print("Yes")
            exit()
        self.update(solution_set, recieved_reward)


    def update(self, solution_set, recieved_reward):
        
        for i in range(len(solution_set)):
            self.T[solution_set[i][1]] += 1
            if self.T[solution_set[i][1]] == 1:
                self.mu_hat[solution_set[i][1]] = recieved_reward[i]
            else:
                self.mu_hat[solution_set[i][1]] = (self.mu_hat[solution_set[i][1]]*(self.T[solution_set[i][1]]-1) + recieved_reward[i])/self.T[solution_set[i][1]]

            if i in self.target_arms and self.mu_hat[solution_set[i][1]] < self.target_reward_min:
                self.target_reward_min = self.mu_hat[solution_set[i][1]]


    def oracle(self):
        
        selected_arms = sorted(self.mu_bar, reverse=True)[:self.args.K]
        return selected_arms


    def model_attack(self, solution_set):

        recieved_reward = []
        for i in solution_set:
            r = min(max(self.mu[i[1]] + np.random.normal(0, self.R, 1)[0], 0), 1)
            if (i[1] in self.target_arms) or (self.T[i[1]] == 0 and r < self.target_reward_min) or (self.T[i[1]] != 0 and (self.mu_hat[i[1]]*self.T[i[1]] + r)/(self.T[i[1]]+1) < self.target_reward_min):
                self.cost += 0
            else:
                self.cost += np.abs(r - self.target_reward_min + self.thresh)
                r = self.target_reward_min - self.thresh
            recieved_reward.append(r)
            self.reward += r

        return recieved_reward