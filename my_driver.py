from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

model = torch.nn.Sequential()
model = torch.load('torcs_ANN')

class MyDriver(Driver):

    # Override the `drive` method to create your own driver
    ...
    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return command

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """

        command = Command()

        speed = carstate.speed_x #np.sqrt(carstate.speed_x**2 + carstate.speed_y**2 + carstate.speed_z**2) ?
        track_position = carstate.distance_from_center
        angle_to_track_axis = carstate.angle
        track_edges = carstate.distances_from_edge

        input = []
        input.append(speed)
        input.append(track_position)
        input.append(angle_to_track_axis)
        for track_edge in track_edges:
            input.append(track_edge)

        print('i', input)

        input = Variable(torch.FloatTensor(input))

        output = model(input.unsqueeze(0))

        output = output.data.numpy()

        print(output)

        self.steer(carstate, 0.0, command)

        ACC_LATERAL_MAX = 6400 * 5
        v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        #print(v_x)
        #v_x = 140

        self.accelerate(carstate, v_x, command)
        #sdfskfsdfssdgfasdfasd
        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command#trainNetwork(x, t):

# Genetic algorithm
# Optimize speed and make it working correctly
# learning rate
# BATCH
# LSTM
# Swarm (?, probably just mention this in paper)
# Work on implementation report
