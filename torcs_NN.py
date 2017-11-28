
import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

torch.manual_seed(1)

import csv

data_matrix_alpine = np.genfromtxt('../train_data/alpine-1.csv', delimiter=',', skip_header=1, skip_footer=1)
data_matrix_aalborg = np.genfromtxt('../train_data/aalborg.csv', delimiter=',', skip_header=1, skip_footer=1)
data_matrix_alpine = np.genfromtxt('../train_data/f-speedway.csv', delimiter=',', skip_header=1, skip_footer=1)
data_matrix = np.concatenate((data_matrix_alpine, data_matrix_aalborg, data_matrix_alpine), axis=0)
data_matrix_shape = data_matrix.shape
data_matrix_input = data_matrix[:,3:]
data_matrix_input_shape = data_matrix_input.shape
data_matrix_output = data_matrix[:,0:3]
data_matrix_output_shape = data_matrix_output.shape

print(data_matrix_shape)

INPUT_DIM = data_matrix_input_shape[1]
HIDDEN_UNITS1 = 300
OUTPUT_DIM = data_matrix_output_shape[1]
#HIDDEN2_UNITS = 600
BATCH_SIZE = 64
ITERATIONS = 300

LEARNING_RATE = 1e-3

#dtype = torch.FloatTensor  # enable CUDA here if you like

input_tensor = Variable(torch.FloatTensor(data_matrix[:, 3:25]), requires_grad = True)
output_tensor = Variable(torch.FloatTensor(data_matrix[:, 0:3]), requires_grad = False)

random.shuffle(data_matrix) # TODO just training data

model = torch.nn.Sequential(
    torch.nn.Linear(INPUT_DIM, HIDDEN_UNITS1),
    torch.nn.Sigmoid(),
    torch.nn.Linear(HIDDEN_UNITS1, HIDDEN_UNITS1),
    torch.nn.Sigmoid(),
    torch.nn.Linear(HIDDEN_UNITS1, OUTPUT_DIM),
)
loss_function = torch.nn.MSELoss(size_average=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# data_matrix_alpine_input_torch = torch.from_numpy(data_matrix_alpine_input)
# data_matrix_alpine_input_torch = data_matrix_alpine_input_torch.float()
# data_matrix_alpine_output_torch = torch.from_numpy(data_matrix_alpine_output)
# data_matrix_alpine_output_torch = data_matrix_alpine_output_torch.float()

# Randomly initialize weights
#w1 = torch.randn(INPUT_DIM, HIDDEN1_UNITS).type(dtype)
#w2 = torch.randn(HIDDEN1_UNITS, OUTPUT_DIM).type(dtype)

#w = Variable(torch.randn(data_matrix_alpine_input.shape[0], data_matrix_alpine_input.shape[1]).type(dtype), requires_grad=True)
#b = Variable(torch.randn(1, data_matrix_alpine_input[1]).type(dtype), requires_grad=True)


for i in range(ITERATIONS):

    output_prediction = model(input_tensor)

    # Compute and print loss.
    loss = loss_function(output_prediction, output_tensor)
    if i % 100 == 0:
        print(i, loss.data[0])

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

print(model(input_tensor[0].unsqueeze(0)))
#model.save_state_dict('torcs_ANN')
torch.save(model, 'torcs_ANN')
