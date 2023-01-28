import numpy as np
import math
import tqdm
import inspect

def tanh2deriv(output):
    return 1-output**2
def softmax(x):
    temp=np.exp(x)
    return temp/np.sum(temp, axis=1, keepdims=True)
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)

def back_propagation(xs, ys, iterations=1000):
    M = xs.shape[1]
    N = ys.shape[1]
    hidden_size = 100
    batch_size = int(iterations/100)
    alpha = 0.01
    weights_0_1 = 0.01*np.random.random((M, hidden_size))
    weights_1_2 = 0.01*np.random.random((hidden_size, N))

    with tqdm.trange(iterations, desc ='progress') as t:
        for j in t:
            for i in range(int(len(xs)/batch_size)):
                #forward feed:
                batch_start, batch_end = (i * batch_size, (i + 1) * batch_size)
                layer_0 = xs[batch_start:batch_end]
                layer_1 = np.tanh(np.dot(layer_0, weights_0_1))
                dropout_mask = np.random.randint(2, size=layer_1.shape)
                layer_1*=dropout_mask
                layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

                # for k in range(batch_size):
                #     delta = ys[batch_start+k]-layer_2[k]
                #     correct_cnt +=np.max([el for el in (delta)])<=0.3
                # t.set_description('correct = '+str(correct_cnt))

                #back_propagation:
                layer_2_delta = sigmoid_deriv(layer_2)*(ys[batch_start:batch_end] - layer_2)
                layer_1_delta = layer_2_delta.dot(weights_1_2.T)*tanh2deriv(layer_1)
                layer_1_delta*=dropout_mask

                weights_1_2+=alpha*layer_1.T.dot(layer_2_delta)
                weights_0_1+=alpha*layer_0.T.dot(layer_1_delta)
            # t.set_description('')
    return (weights_0_1, weights_1_2)

from typing import List
def readable_decoder(line:List[float]):
    outputs =[]
    for idx, el in enumerate(line):
        if el>0.6:
            outputs.append(idx)
    return outputs

def test_network(network, test_xs, test_ys_form,test_ys_noForm,  statistics=True):
    weights_0_1, weights_1_2 =network
    correct_outs=0
    test_correct_cnt = 0
    predicts = []
    for i in range(len(test_xs)):
        layer_0 = test_xs[i]
        layer_1 = np.tanh(np.dot(layer_0, weights_0_1))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
        predicts.append(layer_2)

        pred= readable_decoder(layer_2)
        act =test_ys_noForm[i]

        if statistics:
            delta = [abs(x-y) for x, y in zip(layer_2, test_ys_form[i])]
            print('delta['+str(i)+']:', delta)
            test_correct_cnt += np.max([el for el in (delta)])<=0.3

            print('act_sort = ', list(sorted(act)))
            print('pred = ', pred)

            if i % 10 != 0:
                print('I:' + str(i) + \
                      "Test_Accuracy: " + str(test_correct_cnt / float(len(test_xs))))
            if list(sorted(act)) == pred:
                correct_outs += 1

    print(correct_outs)
    return predicts

# print(*inspect.getargs(tqdm.trange), sep='\n')

