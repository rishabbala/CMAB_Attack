import numpy as np
from math import *
from cucb import CUCB
import argparse
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Arguements for CUCB bandit attack')
parser.add_argument('--feature_dimension', default=5, type=int)
parser.add_argument('--num_arms', default=5, type=int)
parser.add_argument('--K', default=2, type=int) ## num arms to select
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--file_path', default='./CUCB.csv', type=str)
args = parser.parse_args()


if __name__ == '__main__':

    cucb = CUCB(args)
    cucb_attack = CUCB(args, cucb.arms_set, cucb.mu, attack=True)

    for i in range(args.num_iterations):
        print("{}/{}".format(i, args.num_iterations))

        cucb.step()
        cucb_attack.step()

    # plt.plot(range(args.num_iterations), experiment.all_reward, label="exp")
    # plt.plot(range(args.num_iterations), [experiment.optimal_arm_reward]*args.num_iterations, label="opt")
    # plt.legend()
    # plt.show()

    plt.plot(range(args.num_iterations), cucb.cumulative_regret, label="cumulative regret")
    plt.plot(range(args.num_iterations), cucb_attack.cumulative_regret, label="cumulative regret with attack")
    plt.legend()
    plt.show()

    plt.plot(range(args.num_iterations), cucb_attack.cumulative_cost, label="cumulative cost")
    plt.legend()
    plt.show()