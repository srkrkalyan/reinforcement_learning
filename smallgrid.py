#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:33:48 2019

@author: kalyantulabandu
"""

import numpy as np
import mdptoolbox
import matplotlib.pyplot as plt
from modifiedmdp import QLearning as ql

# Environment Creation
# Generates transition probability matrix, rewards matrix each with size AxSxS

rows = 3
cols = 4
actions = {0:'Up',1:'Right',2:'Down',3:'Left'}
walls = [(1,1)]
discounts = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
epsilons = [0.01,0.02,0.05,0.1,0.2,0.5,0.7,0.8,0.9]

counter = 0
state = {}
for i in range(0,rows):
    for j in range(0,cols):
        state[(i,j)] = counter
        counter = counter + 1
T_small = np.zeros((len(actions.keys()),rows*cols,rows*cols))

# Update T for action '0' i.e., 'Up'
for i in range(0,rows):
    for j in range(0,cols):
        # validation for the state (i-1,j)
        if (i,j) not in walls:
            if i-1 in range(0,rows) and j in range(0,cols) and (i-1,j) not in walls:
                T_small[0][state[i,j]][state[(i-1),j]] += 0.8
            else:
                T_small[0][state[i,j]][state[i,j]] += 0.8
           
            # validation for the state (i,j-1)     
            if i in range(0,rows) and j-1 in range(0,cols) and (i,j-1) not in walls:
                T_small[0][state[i,j]][state[i,(j-1)]] += 0.1
            else:
                T_small[0][state[i,j]][state[i,j]] += 0.1
                
            # validation for the state (i,j+1)     
            if i in range(0,rows) and j+1 in range(0,cols) and (i,j+1) not in walls:
                T_small[0][state[i,j]][state[i,(j+1)]] += 0.1
            else:
                T_small[0][state[i,j]][state[i,j]] += 0.1
for each in walls:
    T_small[0][state[each]][state[each]] = 1
    

# Update T for action '1' i.e., 'Right'
for i in range(0,rows):
    for j in range(0,cols):
        # validation for the state (i,j+1)
        if (i,j) not in walls:
            if i in range(0,rows) and j+1 in range(0,cols) and (i,j+1) not in walls:
                T_small[1][state[i,j]][state[i,(j+1)]] += 0.8
            else:
                T_small[1][state[i,j]][state[i,j]] += 0.8
           
            # validation for the state (i+1,j)     
            if i+1 in range(0,rows) and j in range(0,cols) and (i+1,j) not in walls:
                T_small[1][state[(i+1),j]][state[i,j]] += 0.1
            else:
                T_small[1][state[i,j]][state[i,j]] += 0.1
                
            # validation for the state (i-1,j)     
            if i-1 in range(0,rows) and j in range(0,cols) and (i-1,j) not in walls:
                T_small[1][state[(i-1),j]][state[i,j]] += 0.1
            else:
                T_small[1][state[i,j]][state[i,j]] += 0.1
for each in walls:
    T_small[1][state[each]][state[each]] = 1
    
# Update T for action '2' i.e., 'Down'
for i in range(0,rows):
    for j in range(0,cols):
        # validation for the state (i+1,j)
        if (i,j) not in walls:
            if i+1 in range(0,rows) and j in range(0,cols) and (i+1,j) not in walls:
                T_small[2][state[i,j]][state[i+1,j]] += 0.8
            else:
                T_small[2][state[i,j]][state[i,j]] += 0.8
           
            # validation for the state (i,j-1)     
            if i in range(0,rows) and j-1 in range(0,cols) and (i,j-1) not in walls:
                T_small[2][state[i,j]][state[i,(j-1)]] += 0.1
            else:
                T_small[2][state[i,j]][state[i,j]] += 0.1
                
            # validation for the state (i,j+1)     
            if i in range(0,rows) and j+1 in range(0,cols) and (i,j+1) not in walls:
                T_small[2][state[i,j]][state[i,(j+1)]] += 0.1
            else:
                T_small[2][state[i,j]][state[i,j]] += 0.1
for each in walls:
    T_small[2][state[each]][state[each]] = 1
    
# Update T for action '3' i.e., 'Left'
for i in range(0,rows):
    for j in range(0,cols):
        # validation for the state (i,j-1)
        if (i,j) not in walls:
            if i in range(0,rows) and j-1 in range(0,cols) and (i,j-1) not in walls:
                T_small[3][state[i,j]][state[i,(j-1)]] += 0.8
            else:
                T_small[3][state[i,j]][state[i,j]] += 0.8
           
            # validation for the state (i+1,j)     
            if i+1 in range(0,rows) and j in range(0,cols) and (i+1,j) not in walls:
                T_small[3][state[(i+1),j]][state[i,j]] += 0.1
            else:
                T_small[3][state[i,j]][state[i,j]] += 0.1
                
            # validation for the state (i-1,j)     
            if i-1 in range(0,rows) and j in range(0,cols) and (i-1,j) not in walls:
                T_small[3][state[(i-1),j]][state[i,j]] += 0.1
            else:
                T_small[3][state[i,j]][state[i,j]] += 0.1
for each in walls:
    T_small[3][state[each]][state[each]] = 1
    
R_small = np.copy(T_small)

# Setting default reward 

for action in range(0,len(R_small)):
    for row in range(0,len(R_small[action])):
        for column in range(0,len(R_small[action][row])):
            if R_small[action][row][column] > 0:
                R_small[action][row][column] = -0.04

# Add highest reward to absorbing state and lowest reward to be excluded state
R_small[0][state[0,3]][state[0,3]] = 5
R_small[0][state[1,3]][state[1,3]] = -1

R_small[1][state[0,3]][state[0,3]] = 5
R_small[1][state[1,3]][state[1,3]] = -1

R_small[2][state[0,3]][state[0,3]] = 5
R_small[2][state[1,3]][state[1,3]] = -1

R_small[3][state[0,3]][state[0,3]] = 5
R_small[3][state[1,3]][state[1,3]] = -1

# create empty lists to hold Value Iteration results

vi_small_results = {}
pi_small_results = {}
ql_small_results = {}

# Experiment with Value Iteration algorithm
for epsilon in epsilons:
    vi_small_policy = []
    vi_small_iter = []
    vi_small_avgValue = []
    vi_small_time = []
    for each in discounts:
        vi_small = mdptoolbox.mdp.ValueIteration(T_small,R_small,each,epsilon,max_iter=2000)
        vi_small.run()
        vi_small_policy.append(vi_small.policy)
        vi_small_iter.append(vi_small.iter)
        vi_small_avgValue.append(sum([k for k in vi_small.V])/len(vi_small.V))
        vi_small_time.append(vi_small.time)
        vi_small_results[str(epsilon)]= vi_small_policy # index 0
        vi_small_results[str(epsilon)].append(vi_small_iter) # index 1
        vi_small_results[str(epsilon)].append(vi_small_avgValue) #index 2
        vi_small_results[str(epsilon)].append(vi_small_time) # index 3
        
# Experiment with Policy Iteration algorithm
for epsilon in epsilons:
    pi_small_policy = []
    pi_small_iter = []
    pi_small_avgValue = []
    pi_small_time = []
    for each in discounts:
        pi_small = mdptoolbox.mdp.PolicyIterationModified(T_small,R_small,each,epsilon,max_iter=2000)
        pi_small.run()
        pi_small_policy.append(pi_small.policy)
        pi_small_iter.append(pi_small.iter)
        pi_small_avgValue.append(sum([k for k in vi_small.V])/len(pi_small.V))
        pi_small_time.append(pi_small.time)
        pi_small_results[str(epsilon)]= pi_small_policy # index 0
        pi_small_results[str(epsilon)].append(pi_small_iter) # index 1
        pi_small_results[str(epsilon)].append(pi_small_avgValue) #index 2
        pi_small_results[str(epsilon)].append(pi_small_time) # index 3

# visualizing Value Iteration behavior with change in discount rate & epsilon
   
test3 = mdptoolbox.mdp.PolicyIterationModified(T_small,R_small,0.9,0.01,max_iter=2000)
test3.run()
test3.iter
test3.policy
test3.V
test3.time
#print(sum([k for k in test3.V])/len(test3.V))

test4 = mdptoolbox.mdp.ValueIteration(T_small,R_small,0.9,0.01,max_iter=2000)
test4.run()
test4.iter
test4.policy
test4.time
'''

#visualizing Value function for different discount rates
plt.plot([k for k in range(0,12)],test3.V)
plt.plot([k for k in range(0,12)],test4.V)
plt.ylabel('value function')
plt.xlabel('state')
plt.legend(['Max Reward = 2', 'Max Reward = 20'], loc='upper left')
plt.show()
'''
    
plt.plot(discounts,vi_small_results['0.01'][1])
plt.plot(discounts,vi_small_results['0.1'][1])
plt.plot(discounts,vi_small_results['0.9'][1])
plt.title('Small Problem - Value Iteration - Discount Vs No. of iterations')
plt.ylabel('# of iterations')
plt.xlabel('discount rate')
plt.legend(['epsilon = 0.01', 'epsilon = 0.1', 'epsilon = 0.9'], loc='upper left')
plt.show()

plt.plot(discounts,vi_small_results['0.01'][2])
plt.plot(discounts,vi_small_results['0.1'][2])
plt.plot(discounts,vi_small_results['0.9'][2])
plt.title('Small Problem - Value Iteration - Discount Rate Vs Average Value')
plt.ylabel('average value')
plt.xlabel('discount rate')
plt.legend(['epsilon = 0.01', 'epsilon = 0.1', 'epsilon = 0.9'], loc='upper left')
plt.show()

plt.plot(discounts,vi_small_results['0.01'][3])
plt.plot(discounts,vi_small_results['0.1'][3])
plt.plot(discounts,vi_small_results['0.9'][3])
plt.title('Small Problem - Value Iteration - Discount Rate Vs Time')
plt.ylabel('time (in seconds)')
plt.xlabel('discount rate')
plt.legend(['epsilon = 0.01', 'epsilon = 0.1', 'epsilon = 0.9'], loc='upper left')
plt.show()


# visualizing Policy Iteration behavior with change in discount rate & epsilon
    
plt.plot(discounts,pi_small_results['0.01'][1])
plt.plot(discounts,pi_small_results['0.1'][1])
plt.plot(discounts,pi_small_results['0.9'][1])
plt.title('Small Problem - Policy Iteration - Discount Rate Vs No. of iterations')
plt.ylabel('# of iterations')
plt.xlabel('discount rate')
plt.legend(['epsilon = 0.01', 'epsilon = 0.1', 'epsilon = 0.9'], loc='upper left')
plt.show()

plt.plot(discounts,pi_small_results['0.01'][2])
plt.plot(discounts,pi_small_results['0.1'][2])
plt.plot(discounts,pi_small_results['0.9'][2])
plt.ylabel('average value')
plt.xlabel('discount rate')
plt.legend(['epsilon = 0.01', 'epsilon = 0.1', 'epsilon = 0.9'], loc='upper left')
plt.show()

plt.plot(discounts,pi_small_results['0.01'][3])
plt.plot(discounts,pi_small_results['0.1'][3])
plt.plot(discounts,pi_small_results['0.9'][3])
plt.title('Small Problem - Policy Iteration - Discount Rate Vs Time')
plt.ylabel('time (in seconds)')
plt.xlabel('discount rate')
plt.legend(['epsilon = 0.01', 'epsilon = 0.1', 'epsilon = 0.9'], loc='upper left')
plt.show()


ql1 = ql(transitions = T_small,reward = R_small,discount = 0.9, alpha = 0.3, epsilon = 0.5, decay = 1.0)
#ql2 = ql(transitions = T_small,reward = R_small,discount = 0.9, alpha = 0.3, epsilon = 0.6, decay = 1.0)
#ql3 = ql(transitions = T_small,reward = R_small,discount = 0.9, alpha = 0.3, epsilon = 0.7, decay = 1.0)
#ql4 = ql(transitions = T_small,reward = R_small,discount = 0.9, alpha = 0.3, epsilon = 0.4, decay = 1.0)
#ql5 = ql(transitions = T_small,reward = R_small,discount = 0.9, alpha = 0.3, epsilon = 0.2, decay = 1.0)
#iterations = [100,200,300,400,500,600,700,800,900,1000]
ql1.run()
#ql2.run()
#ql3.run()
#ql4.run()
#ql5.run()

'''
plt.plot(iterations,ql1.mean_discrepancy)
plt.plot(iterations,ql2.mean_discrepancy)
plt.plot(iterations,ql3.mean_discrepancy)
plt.plot(iterations,ql4.mean_discrepancy)
#plt.plot(iterations,ql5.mean_discrepancy)
plt.title('Small Problem - Q Learning - Iterations Vs Mean Discrepancy of V')
plt.ylabel('Mean Discrepancy')
plt.xlabel('iterations')
plt.legend(['epsilon = 0.5', 'epsilon = 0.6', 'epsilon = 0.7', 'epsilon = 0.4'], loc='upper right')
plt.show()
'''
# Plot to visualize the value function generated by VI, PI & Q-learning

#plt.plot([k for k in range(0,12)],test4.V)
plt.plot([k for k in range(0,12)],test3.V)
plt.plot([k for k in range(0,12)],ql1.V)
plt.ylabel('value function')
plt.xlabel('state')
plt.title('Small Problem - PI vs Q Learning')
plt.legend(['PI','Q-Learning'], loc='upper right')
plt.show()



