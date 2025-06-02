from learn.toolkit.engine import Value
import random
from typing import List
from typing import List, Union


class Module(object):
    """
    the parent class for all neutal network
    """

    def zero_grad(self):
        """
        This is used to zero out all the gradients of the parameters.
        """
        for p in self.parameters():
            p.grad = 0


    def parameters(self):
        """
        This function is built to be overwritten. This would eventually get us the parameters of the neurons, layers, and the mlp.
        """
        return []

class Neuron(Module):
    """
    A single neuron.
    Parameters:
        number_inputs (int): number of inputs
        is_nonlinear (bool): whether to apply ReLU nonlinearity
        neuron_id (int): um identificador único para este neurônio (ex: "l1_n0" para neurônio 0 da camada 1).

    """
    def __init__(self, number_inputs: int, neuron_id:str, is_nonlinear: bool = True):
        # Create weights for the neuron. The weights are initialized from a random uniform distribution.
        self.weights = [Value(data=random.uniform(-1, 1), label=f"w_n{neuron_id}_i{idx}") for idx, _ in enumerate(range(number_inputs), start=1)]
        # Create bias for the neuron.
        self.bias = Value(data=0.0, label=f"b_n{neuron_id}")
        self.is_nonlinear = is_nonlinear
        self.neuron_id = neuron_id

    def __call__(self, x: List["Value"]) -> "Value":
        # Calcula a soma ponderada + bias
            # w.x + b
        # Compute the dot product of the input and the weights. 
        # Add the bias to the dot product.

        act = sum(
            ((wi * xi) for wi, xi in zip(self.weights, x)),
            self.bias
        )
        # If activation is mentioned, apply ReLU to it.
        act.label = f"act_n{self.neuron_id}"
        return act.relu() if self.is_nonlinear else act # Aplica tanh ou retorna linear

    def parameters(self):
        # Get the parameters of the neuron. The parameters of a neuron
        # is its weights and bias.
        return self.weights + [self.bias]

    def __repr__(self):
        # Print a better representation of the neuron.
        non_linearity_type = 'ReLU' if self.is_nonlinear else 'Linear'
        return f"Neuron(id={self.neuron_id}, inputs={len(self.weights)}, Act={non_linearity_type})"
        # return f"Neuron {self.neuron_id}(Number={len(self.weights)}, Non-Linearity={'ReLU' if self.is_nonlinear else 'None'})"


class Layer(Module):
    """
    A layer of neurons.
    Parameters:
        number_inputs (int): number of inputs
        number_outputs (int): number of outputs
        layer_id (int): um identificador único para esta camada (ex: "l1").

    """
    def __init__(self, number_inputs: int, number_outputs: int, layer_id: int, **kwargs):
        # A layer is a list of neurons.
        self.neurons = [
            Neuron(number_inputs=number_inputs, 
                   neuron_id=str(idx), 
                   **kwargs) for idx in range(number_outputs)
        ]
        self.layer_id = layer_id
        self.number_outputs = number_outputs

    def __call__(self, x: List["Value"]) -> Union[List["Value"], "Value"]:
        # Iterate over all the neurons and compute the output of each.
        outs = [n(x) for n in self.neurons]

        return outs if self.number_outputs != 1 else outs[0] # Retorna único valor ou lista

    def parameters(self):
        # The parameters of a layer is the parameters of all the neurons.
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        # Print a better representation of the layer.
        neuron_details = [f"    - {neuron!r}" for neuron in self.neurons]
        return f"Layer(id={self.layer_id}, Neurons:\n" + "\n".join(neuron_details) + "\n  )"

class MLP(Module):
    """
    The Multi-Layer Perceptron (MLP) class.
    Parameters:
        number_inputs (int): number of inputs.
        list_number_outputs (List[int]): number of outputs in each layer.
    """
    def __init__(self, number_inputs: int, list_number_outputs: List[int]):
        # Get the number of inputs and all the number of outputs in a single list.

        # Tamanhos das camadas: [n_inputs, n_hidden1, n_hidden2, ..., n_output]
        total_size = [number_inputs] + list_number_outputs

        # Build layers by connecting each layer to the previous one.
        self.layers = [
            Layer(
                number_inputs=total_size[i],          # Entradas para a camada atual
                number_outputs=total_size[i + 1],     # Neurônios (saídas) na camada atual
                layer_id=f"l{idx}",                   # ID da camada (ex: "l0", "l1")
                is_nonlinear= (i != len(list_number_outputs) - 1) # Não linearidade, exceto na última camada
            )
            for idx, i in enumerate(range(len(list_number_outputs)))
        ]

        # A última camada geralmente é linear (is_nonlinear=False) antes da loss,
        # ou a não-linearidade é parte da função de loss (ex: Softmax com CrossEntropy)
        # No micrograd, é comum aplicar tanh até na última camada se for uma saída genérica.

    def __call__(self, x: List["Value"]) -> List["Value"]:
        # Iterate over the layers and compute the output of
        # each sequentially.
        current_input = x
        for layer in self.layers:
            current_input = layer(current_input)
            
        return current_input
    
    def parameters(self):
        # Get the parameters of the MLP
        # return [layer for layer in self.layers for layer in layer.parameters()]
        return [p for layer_obj in self.layers for p in layer_obj.parameters()]
    
    
    def __repr__(self):
        # Print a better representation of the MLP.
        layer_reprs = [f"  - Layer {layer.layer_id} ({len(layer.neurons[0].weights)} inputs, {len(layer.neurons)} neurons)" for layer in self.layers]
        return f"MLP of {len(self.layers)} layers:\n" + "\n".join(layer_reprs)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

# x = [2.0, 3.0]
# mlp = MLP(number_inputs=2, list_number_outputs=[3, 3, 1])
# print(mlp)
# out = mlp(x)
# print(f"Output => {out}")
    
# x = [2.0, 3.0]
# layer = Layer(number_inputs=2, number_outputs=3, name=1)
# print(layer)
# out = layer(x)
# print(f"Output => {out}")
# print()

# x = [2.0, 3.0]
# layer = Layer(number_inputs=2, number_outputs=1, name=1)
# print(layer)
# out = layer(x)
# print(f"Output => {out}")

# x = [2.0, 3.0]
# neuron = Neuron(number_inputs=2, name=1)
# print(neuron)
# out = neuron(x)
# print(f"Output => {out}")