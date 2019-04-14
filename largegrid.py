#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:07:39 2019

@author: kalyantulabandu
"""
import numpy as np
import mdptoolbox
import matplotlib.pyplot as plt
from modifiedmdp import QLearning as ql

# Environment creation
rows = 20
cols = 20
actions = {0:'Up',1:'Right',2:'Down',3:'Left'}
walls = [(1,18),(2,17),(3,16),(4,15),(5,14),(6,5),(8,8),(8,3),(17,2),(13,2),(14,3),(15,4),(16,5),(17,6),(16,7),(15,8),(14,9),(13,10),(14,11),(15,12),(14,13),(14,14)]
discounts = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
epsilons = [0.01,0.02,0.05,0.1,0.2,0.5,0.7,0.8,0.9]

counter = 0
state = {}
for i in range(0,rows):
    for j in range(0,cols):
        state[(i,j)] = counter
        counter = counter + 1
sample = np.zeros((len(actions.keys()),rows*cols,rows*cols))

# Update T for action '0' i.e., 'Up'
for i in range(0,rows):
    for j in range(0,cols):
        # validation for the state (i-1,j)
        if (i,j) not in walls:
            if i-1 in range(0,rows) and j in range(0,cols) and (i-1,j) not in walls:
                sample[0][state[i,j]][state[(i-1),j]] += 0.8
            else:
                sample[0][state[i,j]][state[i,j]] += 0.8
           
            # validation for the state (i,j-1)     
            if i in range(0,rows) and j-1 in range(0,cols) and (i,j-1) not in walls:
                sample[0][state[i,j]][state[i,(j-1)]] += 0.1
            else:
                sample[0][state[i,j]][state[i,j]] += 0.1
                
            # validation for the state (i,j+1)     
            if i in range(0,rows) and j+1 in range(0,cols) and (i,j+1) not in walls:
                sample[0][state[i,j]][state[i,(j+1)]] += 0.1
            else:
                sample[0][state[i,j]][state[i,j]] += 0.1
for each in walls:
    sample[0][state[each]][state[each]] = 1
    

# Update T for action '1' i.e., 'Right'
for i in range(0,rows):
    for j in range(0,cols):
        # validation for the state (i,j+1)
        if (i,j) not in walls:
            if i in range(0,rows) and j+1 in range(0,cols) and (i,j+1) not in walls:
                sample[1][state[i,j]][state[i,(j+1)]] += 0.8
            else:
                sample[1][state[i,j]][state[i,j]] += 0.8
           
            # validation for the state (i+1,j)     
            if i+1 in range(0,rows) and j in range(0,cols) and (i+1,j) not in walls:
                sample[1][state[(i+1),j]][state[i,j]] += 0.1
            else:
                sample[1][state[i,j]][state[i,j]] += 0.1
                
            # validation for the state (i-1,j)     
            if i-1 in range(0,rows) and j in range(0,cols) and (i-1,j) not in walls:
                sample[1][state[(i-1),j]][state[i,j]] += 0.1
            else:
                sample[1][state[i,j]][state[i,j]] += 0.1
for each in walls:
    sample[1][state[each]][state[each]] = 1
    
# Update T for action '2' i.e., 'Down'
for i in range(0,rows):
    for j in range(0,cols):
        # validation for the state (i+1,j)
        if (i,j) not in walls:
            if i+1 in range(0,rows) and j in range(0,cols) and (i+1,j) not in walls:
                sample[2][state[i,j]][state[i+1,j]] += 0.8
            else:
                sample[2][state[i,j]][state[i,j]] += 0.8
           
            # validation for the state (i,j-1)     
            if i in range(0,rows) and j-1 in range(0,cols) and (i,j-1) not in walls:
                sample[2][state[i,j]][state[i,(j-1)]] += 0.1
            else:
                sample[2][state[i,j]][state[i,j]] += 0.1
                
            # validation for the state (i,j+1)     
            if i in range(0,rows) and j+1 in range(0,cols) and (i,j+1) not in walls:
                sample[2][state[i,j]][state[i,(j+1)]] += 0.1
            else:
                sample[2][state[i,j]][state[i,j]] += 0.1
for each in walls:
    sample[2][state[each]][state[each]] = 1
    
# Update T for action '3' i.e., 'Left'
for i in range(0,rows):
    for j in range(0,cols):
        # validation for the state (i,j-1)
        if (i,j) not in walls:
            if i in range(0,rows) and j-1 in range(0,cols) and (i,j-1) not in walls:
                sample[3][state[i,j]][state[i,(j-1)]] += 0.8
            else:
                sample[3][state[i,j]][state[i,j]] += 0.8
           
            # validation for the state (i+1,j)     
            if i+1 in range(0,rows) and j in range(0,cols) and (i+1,j) not in walls:
                sample[3][state[(i+1),j]][state[i,j]] += 0.1
            else:
                sample[3][state[i,j]][state[i,j]] += 0.1
                
            # validation for the state (i-1,j)     
            if i-1 in range(0,rows) and j in range(0,cols) and (i-1,j) not in walls:
                sample[3][state[(i-1),j]][state[i,j]] += 0.1
            else:
                sample[3][state[i,j]][state[i,j]] += 0.1
for each in walls:
    sample[3][state[each]][state[each]] = 1
    
R = np.copy(sample)

for action in range(0,len(R)):
    for row in range(0,len(R[action])):
        for column in range(0,len(R[action][row])):
            if R[action][row][column] > 0:
                R[action][row][column] = -0.04

# Add highest reward to absorbing state and lowest reward to be excluded state
R[0][state[0,19]][state[0,19]] = 15
R[0][state[1,19]][state[1,19]] = -4

R[1][state[0,19]][state[0,19]] =15
R[1][state[1,19]][state[1,19]] = -4

R[2][state[0,19]][state[0,19]] = 15
R[2][state[1,19]][state[1,19]] = -4

R[3][state[0,19]][state[0,19]] = 15
R[3][state[1,19]][state[1,19]] = -4

# create empty lists to hold Value Iteration results

vi_results = {}
pi_results = {}
ql_results = {}

# Experiment with Value Iteration algorithm
for epsilon in epsilons:
    vi_policy = []
    vi_iter = []
    vi_avgValue = []
    vi_time = []
    for each in discounts:
        vi_large = mdptoolbox.mdp.ValueIteration(sample,R,each,epsilon,max_iter=2000)
        vi_large.run()
        vi_policy.append(vi_large.policy)
        vi_iter.append(vi_large.iter)
        vi_avgValue.append(sum([k for k in vi_large.V])/len(vi_large.V))
        vi_time.append(vi_large.time)
        vi_results[str(epsilon)]= vi_policy # index 0
        vi_results[str(epsilon)].append(vi_iter) # index 1
        vi_results[str(epsilon)].append(vi_avgValue) #index 2
        vi_results[str(epsilon)].append(vi_time) # index 3
        
# Experiment with Value Iteration algorithm
for epsilon in epsilons:
    pi_policy = []
    pi_iter = []
    pi_avgValue = []
    pi_time = []
    for each in discounts:
        pi_large = mdptoolbox.mdp.PolicyIterationModified(sample,R,each,epsilon,max_iter=2000)
        pi_large.run()
        pi_policy.append(pi_large.policy)
        pi_iter.append(pi_large.iter)
        pi_avgValue.append(sum([k for k in vi_large.V])/len(pi_large.V))
        pi_time.append(pi_large.time)
        pi_results[str(epsilon)]= pi_policy # index 0
        pi_results[str(epsilon)].append(pi_iter) # index 1
        pi_results[str(epsilon)].append(pi_avgValue) #index 2
        pi_results[str(epsilon)].append(pi_time) # index 3

# visualizing Value Iteration behavior with change in discount rate & epsilon

test1 = mdptoolbox.mdp.ValueIteration(sample,R,0.9,0.01,max_iter=2000)
test1.run()
test1.iter
#print(sum([k for k in test1.V])/len(test1.V))
test1.V

test2 = mdptoolbox.mdp.PolicyIterationModified(sample,R,0.9,0.01,max_iter=2000)
test2.run()
test2.iter
test2.V 
'''
#visualizing Value function for different discount rates
plt.plot([k for k in range(0,11)],test1.V)
plt.plot([k for k in range(0,11)],test2.V)
plt.ylabel('value function')
plt.xlabel('state')
plt.legend(['Max Reward = 30', 'Max Reward = 5'], loc='upper left')
plt.show()
'''

    
plt.plot(discounts,vi_results['0.01'][1])
plt.plot(discounts,vi_results['0.1'][1])
plt.plot(discounts,vi_results['0.9'][1])
plt.title('Large Problem - Value Iteration - Discount Vs No. of iterations')
plt.ylabel('# of iterations')
plt.xlabel('discount rate')
plt.legend(['epsilon = 0.01', 'epsilon = 0.1', 'epsilon = 0.9'], loc='upper left')
plt.show()

plt.plot(discounts,vi_results['0.01'][2])
plt.plot(discounts,vi_results['0.1'][2])
plt.plot(discounts,vi_results['0.9'][2])
plt.title('Large Problem - Value Iteration - Discount Rate Vs Average Value')
plt.ylabel('average value')
plt.xlabel('discount rate')
plt.legend(['epsilon = 0.01', 'epsilon = 0.1', 'epsilon = 0.9'], loc='upper left')
plt.show()

plt.plot(discounts,vi_results['0.01'][3])
plt.plot(discounts,vi_results['0.1'][3])
plt.plot(discounts,vi_results['0.9'][3])
plt.title('Large Problem - Value Iteration - Discount Rate Vs Time')
plt.ylabel('time (in seconds)')
plt.xlabel('discount rate')
plt.legend(['epsilon = 0.01', 'epsilon = 0.1', 'epsilon = 0.9'], loc='upper left')
plt.show()


# visualizing Policy Iteration behavior with change in discount rate & epsilon
    
plt.plot(discounts,pi_results['0.01'][1])
plt.plot(discounts,pi_results['0.1'][1])
plt.plot(discounts,pi_results['0.9'][1])
plt.title('Large Problem - Policy Iteration - Discount Rate Vs No. of iterations')
plt.ylabel('# of iterations')
plt.xlabel('discount rate')
plt.legend(['epsilon = 0.01', 'epsilon = 0.1', 'epsilon = 0.9'], loc='upper left')
plt.show()

plt.plot(discounts,pi_results['0.01'][2])
plt.plot(discounts,pi_results['0.1'][2])
plt.plot(discounts,pi_results['0.9'][2])
plt.title('Large Problem - Policy Iteration - Discount Rate Vs Average Value')
plt.ylabel('average value')
plt.xlabel('discount rate')
plt.legend(['epsilon = 0.01', 'epsilon = 0.1', 'epsilon = 0.9'], loc='upper left')
plt.show()

plt.plot(discounts,pi_results['0.01'][3])
plt.plot(discounts,pi_results['0.1'][3])
plt.plot(discounts,pi_results['0.9'][3])
plt.title('Large Problem - Policy Iteration - Discount Rate Vs Time')
plt.ylabel('time (in seconds)')
plt.xlabel('discount rate')
plt.legend(['epsilon = 0.01', 'epsilon = 0.1', 'epsilon = 0.9'], loc='upper left')
plt.show()

ql1 = ql(transitions = sample,reward = R,discount = 0.9, alpha = 0.3, epsilon = 0.5, decay = 1.0)
iter = ql1.run()
ql_result = ql1.Q
ql1.policy
len(ql1.V)
len(ql1.mean_discrepancy)

ql2 =  mdptoolbox.mdp.QLearning(sample,R,0.9)
ql2.run()
Qtest = ql1.Q

plt.plot([k for k in range(0,400)],ql2.Q[ :,2])
#plt.plot([k for k in range(0,400)],test1.V)
plt.plot([k for k in range(0,400)],ql1.Q[ :,2])
plt.ylabel('Q Value')
plt.xlabel('state')
plt.title('Large Problem - Q Learning (Action 2) - Without and with exploration strategy')
plt.legend(['QL - No exploration strategy','QL - Epsilon Greedy Strategy'], loc='upper right')
plt.show()