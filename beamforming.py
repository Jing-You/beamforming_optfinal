# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:22:13 2020

@author: User
"""


from gurobipy import *

import numpy as np

np.random.seed(32)


FDD = 8
num_user = 360
time_interval = 10
beam_dir = 60
num_beam = 1

# Initialize

Xposition = np.random.rand(num_user, num_user)-0.5

X_id = np.zeros([num_user, beam_dir], dtype="bool")

for i in range(num_user):
    phi = np.arctan2(Xposition[i][1], Xposition[i][0])+np.pi
    # print(phi, phi // (2*np.pi / beam_dir))
    X_id[i][int(phi // (2*np.pi / beam_dir))] = 1  



H_i = 1 - np.sqrt(Xposition[:, 0]**2 + Xposition[:, 1]**2)/((0.5**2+0.5**2)**0.5)
D = np.random.randint(low=1, high=10, size=num_user)

D_rec = 1.0 / D


# Create a new model
m = Model("beamforming")

# Create variables
X_it = {}
for i in range(num_user):
    for t in range(time_interval):
        X_it[i, t] = m.addVar(vtype=GRB.BINARY)


B_dt = {}
for p in range(beam_dir):
    for t in range(time_interval):
        B_dt[p, t] = m.addVar(vtype=GRB.BINARY)


# Integrate new variables
m.update()



# Set objective
m.setObjective((quicksum(X_it[i,t]*H_i[i]for i in range(num_user) for t in range(time_interval)))*D_rec[i], GRB.MAXIMIZE) 

for i in range(num_user):
    m.addConstr(quicksum(X_it[i,t] for t in range(time_interval)) <= D[i])

for t in range(time_interval):
    m.addConstr(quicksum(B_dt[d,t] for d in range(beam_dir)) <= 1)

for t in range(time_interval):
    m.addConstr(quicksum(X_it[i,t] for i in range(num_user)) <= FDD)


for i in range(num_user):
    for t in range(time_interval):
        m.addConstr(X_it[i,t]*quicksum(X_id[i, p]*B_dt[p, t] for p in range(beam_dir)) == 1*quicksum(X_id[i, p]*B_dt[p, t] for p in range(beam_dir)))


for i in range(num_user):
    for t in range(time_interval):
        m.addConstr(X_it[i,t]<=quicksum(X_id[i, p]*B_dt[p, t] for p in range(beam_dir)))

m.optimize()



print('Obj: %g' % m.objVal)

X = np.empty([num_user, time_interval])
for user in range(num_user):
    for t in range(time_interval):
        X[user, t]=X_it[user, t].X


print(X)