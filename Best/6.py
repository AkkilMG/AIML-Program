import numpy as np

x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) / np.amax(np.array(([2, 9], [1, 5], [3, 6]), dtype=float), axis=0)
y = np.array(([92], [86], [89]), dtype=float) / 100

sigmoid = lambda x: 1 / (1 + np.exp(-x))
d_sigmoid = lambda x: x * (1 - x)

epoch, lr = 5000, 0.1
input_neurons, hidden_neurons, output_neurons = 2, 3, 1
wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
wout = np.random.uniform(size=(hidden_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

for _ in range(epoch):
    hinp = np.dot(x, wh) + bh
    hlayer_act = sigmoid(hinp)
    outinp = np.dot(hlayer_act, wout) + bout
    output = sigmoid(outinp)
    eo = y - output
    d_output = eo * d_sigmoid(output)
    eh = d_output.dot(wout.T)
    d_hiddenlayer = eh * d_sigmoid(hlayer_act)
    wout += hlayer_act.T.dot(d_output) * lr
    wh += x.T.dot(d_hiddenlayer) * lr

print(f"Input:\n{x}")
print(f"Actual Output:\n{y}")
print(f"Predicted Output:\n{output}")

# OUTPUT:
# Input:
# [[0.66666667 1.        ]
#  [0.33333333 1.        ]
#  [1.         0.66666667]]
# Actual:
# [[0.92]
#  [0.86]
#  [0.89]]
# Predicted:
#  [[0.89155219]
#  [0.88386888]
#  [0.89127146]]
