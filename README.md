# ac_snn

This is an implementation of Actor-Critic algorithm on spiking neural network using temporal coding method.

The neuron model is in neuron6.py, the net architecture is in net3-3-3.py. This net is based on a 5 * 5 grid map, and a new problem needs a new net, which means you can not use net3-3-3.py directly on other problems. However, those update methods in net3-3-3.py can be used for reference while a new net is formed.

Always run init2-2.py at first and then run net3-3-3.py. A result5.py is provided, take it as an example to use the network to implement decision making.

Junqi Lu
10.21.2021
