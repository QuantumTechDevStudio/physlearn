# Необходмые команды импорта.
import sys
sys.path.append('../physlearn/')
from physlearn.NeuralNet.NeuralNet import NeuralNet
from physlearn.NeuralNet.SubNet import SubNet
import tensorflow as tf

# Returns net, net graph, graph for sum of net outputs, session
def return_net_expressions(num_outs, num_sig):
    net = NeuralNet(-2,2)
    net.add_input_layer(1)
    net.add(num_sig, tf.sigmoid)
    net.add_output_layer(num_outs, net.linear)
    net.compile()
    net.enable_numpy_mode()
    return net, net.return_graph(), tf.reduce_sum(input_tensor = net.return_graph(), axis = 0), net.return_session()

# Returns deep net, net graph, graph for sum of net outputs, session
def return_deep_net_expressions(num_outs, num_hid1, num_hid2):
    net = NeuralNet(-2,2)
    net.add_input_layer(1)
    net.add(num_hid1, gaussian)
    net.add(num_hid2, tf.sigmoid)
    net.add_output_layer(num_outs, net.linear)
    net.compile()
    net.enable_numpy_mode()
    return net, net.return_graph(), tf.reduce_sum(input_tensor = net.return_graph(), axis = 0), net.return_session()

# Returns separated deep net, net graph, graph for sum of net outputs, session
def return_separated_deep_net_expressions(num_outs, num_hid1, num_hid2):
    net = NeuralNet(-2,2)
    net.add_input_layer(1)
    sub_nets = []
    for i in range(num_outs):
        sub_nets.append(SubNet())
        sub_nets[i].add(num_hid1)
        sub_nets[i].add(num_hid2)
        sub_nets[i].add_output_layer(1)
    net.add_sub_nets(sub_nets,  [tf.sigmoid, tf.sigmoid, net.linear]) #
    net.compile()
    net.enable_numpy_mode()
    return net, net.return_graph(), tf.reduce_sum(input_tensor = net.return_graph(), axis = 0), net.return_session()

# Returns separated deep net, net graph, graph for sum of net outputs, session
def return_separated_net_expressions(num_outs, num_hid):
    net = NeuralNet(-2,2)
    net.add_input_layer(1)
    sub_nets = []
    for i in range(num_outs):
        sub_nets.append(SubNet())
        sub_nets[i].add(num_hid)
        sub_nets[i].add_output_layer(1)
    net.add_sub_nets(sub_nets,  [tf.sigmoid, net.linear]) #
    net.compile()
    net.enable_numpy_mode()
    return net, net.return_graph(), tf.reduce_sum(input_tensor = net.return_graph(), axis = 0), net.return_session()

def gaussian(x):
	return tf.exp(-tf.square(x))



















