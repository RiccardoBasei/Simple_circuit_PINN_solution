import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
import torch
import torch.nn as nn
import torch.optim as optim
import circuit_simulator as cs
from net import Net
import time
import matlab.engine

# data creation
netlist = {'starting nodes': np.array([3, 0, 1, 2, 0, 1, 2]),
           'ending nodes': np.array([0, 1, 2, 3, 3, 3, 3]),
           'types': np.array(['I', 'R', 'R', 'R', 'C', 'C', 'C']),
           'values': np.array([1, 1, 1, 10, 0.001, 0.001, 0.001])}

# source and time interval data
freq = 2000
t_start = 0.0
t_end = 3.5/freq
dt = t_end/5000
t = np.arange(t_start+dt/1e10, t_end, dt, dtype=float)
t = np.concatenate((t_start, t), axis=None)
i = lambda tt: 10.0*(np.sin(2*np.pi*freq*tt))
# i = lambda tt: 5.0*(1-np.exp(tt*freq))

i_t = i(t)

# number of edges
e = len(netlist['types'])

# number of nodes
n = int(np.max([np.max(netlist['starting nodes']),np.max(netlist['ending nodes'])])+1)

# incidence matrix
A = cs.buildA(netlist['starting nodes'], netlist['ending nodes'], n, e)

# indexes vector and matrices building
index_R, index_Vs, index_Is, index_L, index_C, index_D = cs.reading_net(netlist['types'])
vecGen = np.concatenate((index_Vs, index_Is))
R, G, L, C, s = cs.build_matrices(netlist['types'], netlist['values'], e, index_R, index_Vs, index_Is, index_L, index_C, index_D)

# nodal potential matrices
M1, M2 = cs.nodal_analysis(A, G, C)

# initial conidition and RK solution
x0 = np.zeros((1,n-1)).flatten()
start = time.time()
x = cs.theta_method(M1, M2, 0.7, x0, dt, t, i_t, netlist['values'], vecGen, e, n, A, formulation='NA')
end = time.time()
teul = end-start

# normalization
t_norm =t/t_end
x_norm = x

# Training data
n_train_points = 0
# train_indexes = randint(0, t.size, n_train_points)
# t_train = t_norm[train_indexes]
# x_train = x_norm[:,train_indexes]

### Validation data
n_val_points = 500
val_indexes = randint(0, t.size, n_val_points)
t_val = t_norm[val_indexes]
x_val = x_norm[:,val_indexes]

# Collocation points
n_collocations = 1000
collocation_index = np.linspace(0, t.size-1, n_collocations, dtype = int)
t_collocations_norm = t_norm[collocation_index]
t_collocations_norm = torch.reshape(torch.tensor(t_collocations_norm).float(), (len(collocation_index),1)).requires_grad_()

# data preparation: torch tensors creation
# t_train = torch.reshape(torch.Tensor(t_train).float(), (len(train_indexes),1))
# x_train = torch.transpose(torch.Tensor(x_train).float(), 0, 1)
t_val = torch.reshape(torch.Tensor(t_val).float(), (len(val_indexes),1))
x_val = torch.transpose(torch.Tensor(x_val).float(), 0, 1)
x0 = torch.tensor(x_norm[:,0]).float()

M1 = torch.Tensor(M1)
M2 = torch.Tensor(M2)

# Neural Network definition
layers = [1, 5, 40, 40, 40, 40, 40, n-1]
net = Net(layers)

torch.manual_seed(0)
print('Network initialized')

# training parameters
num_epochs = 6000
optimizer = optim.Adam(net.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 'min',
                                                 factor=0.7,
                                                 cooldown=25,
                                                 patience=90)
loss_fn = nn.MSELoss()

# implementation of the physical loss function
def get_grad(net, inp):

    out = net(inp)
    out = torch.transpose(out,0,1)
    out_real = torch.zeros(out.shape)
    out_t = torch.zeros(out.shape)
    for idx in range(n-1):
        out_real[idx,:] = out[idx,:]
        out_t[idx,:] = 1/t_end * torch.autograd.grad(out_real[idx,:],
                                                     inp,
                                                     grad_outputs = torch.ones_like(out_real[idx,:]),
                                                     create_graph=True)[0].flatten()
        
    return out_t, out_real

rhs_matrix=torch.Tensor([0,0,0])
for nnn in range(n_collocations):
    netlist['values'][vecGen] = i(t_collocations_norm[nnn].detach().numpy()*t_end)
    rhs_true = torch.Tensor(cs.assemblerhs(vecGen, netlist['values'], n, e, A, formulation='NA'))[:,0]
    rhs_matrix = torch.vstack([rhs_matrix,rhs_true])
rhs_matrix = torch.transpose(rhs_matrix[1:,:],0,1)

def f(net, inp, M1, M2, rhs_matrix):

    out_t, out_real = get_grad(net, inp)
    t = inp
    lhs_model = torch.matmul(M1, out_real) + torch.matmul(M2, out_t)
    physics_res =lhs_model - rhs_matrix
    
    return physics_res

start = time.time()
# TRAINING LOOP
train_loss_log = []
val_loss_log = []
for epoch_num in range(num_epochs):

    print('#################')
    print(f'# EPOCH {epoch_num}')
    print('#################')

    ### TRAIN
    train_loss = []
    net.train()
    
    # Forward pass
    # out_data = net(t_train)
    phys_res = f(net, t_collocations_norm, M1, M2, rhs_matrix)

    # Compute loss
    # mse_data = loss_fn(out_data, x_train)
    mse_phys = loss_fn(phys_res, torch.zeros(phys_res.shape))
    mse_ci = loss_fn(net(torch.Tensor([0.])), x0)
    loss = 0.1*mse_phys + mse_ci #+ mse_data

    # Backpropagation
    net.zero_grad()
    loss.backward()

    # Update the weights and biases
    optimizer.step()

    # Save train loss for this batch
    loss_batch = loss.detach().numpy()
    train_loss.append(loss_batch)

    # Save average train loss
    train_loss = np.mean(train_loss)
    print(f"AVERAGE TRAIN LOSS: {train_loss}")
    # print(f"DATA LOSS: {mse_data}")
    print(f"PHYSICS LOSS: {mse_phys}")
    print(f"C.I. LOSS: {mse_ci}")
    train_loss_log.append(train_loss)

    ### VALIDATION
    val_loss= []
    net.eval() # Evaluation mode (e.g. disable dropout, batchnorm,...)
    with torch.no_grad(): # Disable gradient tracking

        # Forward pass
        out = net(t_val)
        
        # Compute loss
        loss = loss_fn(out, x_val)

        # Save val loss for this batch
        loss_batch = loss.detach().numpy()
        val_loss.append(loss_batch)

    # Save average validation loss
    val_loss = np.mean(val_loss)
    print(f"AVERAGE VAL LOSS: {np.mean(val_loss)}")
    val_loss_log.append(val_loss)
    
    scheduler.step(loss)
end = time.time()
tPINN = end - start

# evaluation and plots
x_model = net(t_val)
x_model = x_model.detach().numpy()

# error evaluation with different methods
t = np.arange(t_start+dt/1e10+dt, t_end, dt, dtype=float)
eng = matlab.engine.start_matlab()
p1, p2, p3 = eng.analytical_solution(t, nargout=3)
p1 = np.array(p1)
p2 = np.array(p2)
p3 = np.array(p3)
x = np.hstack([p1,p2,p3])

# theta method error
err = np.abs(x.T - x_norm[:,2:])
MSEeul = np.sqrt(np.mean(err**2))
MAXeul = np.max(err)

# PINN method error
t_tensor = torch.reshape(torch.Tensor(t/t_end).float(), (4999,1))
x_model1 = net(t_tensor)
x_model1 = x_model1.detach().numpy()
err = np.abs(x - x_model1)
MSEPINN = np.sqrt(np.mean(err**2))
MAXPINN = np.max(err)

plt.figure(1)
plt.plot(t*1000, p1)
plt.plot(t*1000, x_model1[:,0], '--')
plt.xlabel('t [ms]')
plt.ylabel('p1 [V]')
plt.legend(['analytical','PINN'])
plt.grid(linewidth=0.5)
#plt.savefig('circuital_problem/p1.png')

plt.figure(2)
plt.plot(t*1000, p2)
plt.plot(t*1000, x_model1[:,1], '--')
plt.xlabel('t [ms]')
plt.ylabel('p2 [V]')
plt.legend(['analytical','PINN'])
plt.grid(linewidth=0.5)
#plt.savefig('circuital_problem/p2.png')
         
plt.figure(3)
plt.plot(t*1000, p3)
plt.plot(t*1000, x_model1[:,2], '--')
plt.xlabel('t [ms]')
plt.ylabel('p3 [V]')
plt.legend(['analytical','PINN'])
plt.grid(linewidth=0.5)
#plt.savefig('circuital_problem/p3.png')

plt.figure(4)
plt.semilogy(range(num_epochs),train_loss_log)
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.grid(linewidth=0.5)
#plt.savefig('circuital_problem/loss.png')

# print results
print('')
print('theta method')
print(f' solution time: {teul} s')
print(f' MSE: {MSEeul} V, MAX: {MAXeul} V')
print('')
print('PINN')
print(f' solution time: {tPINN} s')
print(f' MSE: {MSEPINN} V, MAX: {MAXPINN} V')

#show all the figures
plt.show()