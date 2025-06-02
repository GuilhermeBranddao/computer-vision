import random
import math
import numpy as np 

import math  # Para funções matemáticas como exp, log, tanh
from typing import List, Union

class Value:
    """
    Representa um valor escalar que participa de um grafo computacional
    para permitir a diferenciação automática (autograd).

    Attributes:
        data (float): The data for the Value node.
        _children (Tuple): The children of the current node.
    """
    
    def __init__(self, data: float, _children: tuple = (), _op: str = '', label: str = ''):
        """
        Inicializa um objeto Value.

        Args:
            data (float): O valor numérico escalar.
            _children (tuple, optional): Uma tupla de objetos Value que são os "filhos"
                                         (operandos) que geraram este Value. Default é ().
            _op (str, optional): A operação que gerou este Value (ex: '+', '*'). Default é ''.
            label (str, optional): Um rótulo opcional para este Value, útil para depuração. Default é ''.
        """
        self.data = data

        # The partial gradient of the last node with respect to this
        # node. This is also termed as the global gradient.
        # Gradient 0.0 means that there is no effect of the change
        # of the last node with respect to this node. On
        # initialization it is assumed that all the variables have no
        # effect on the entire architecture.

        self.grad = 0.0  # Gradiente da saída final da expressão em relação a este Value.]
        
        # The function that derives the gradient of the children nodes
        # of the current node. It is easier this way, because each node
        # is built from children nodes and an operation. Upon back-propagation
        # the current node can easily fill in the gradients of the children.
        # Note: The global gradient is the multiplication of the local gradient
        # and the flowing gradient from the parent.
        # Atributos internos para o backpropagation e construção do grafo:
        self._backward = lambda: None  # Função que propaga o gradiente para os filhos.]
                                      # Será sobrescrita por cada operação.
        
        # Define the children of this node.
        self._prev = set(_children)   # Conjunto dos nós filhos (operandos).]
        self._op = _op                # Operação que criou este nó.]
        self.label = label            # Rótulo para identificação.]

        # try:
        #     # FrameInfo(frame, filename, lineno, function, code_context, index)
        #     self.func = inspect.stack()[1].function
        # except IndexError:
        #     # Caso seja instanciado no escopo global ou de forma inesperada
        #     self.func = "<escopo global ou desconhecido>"

    def custom_addition(self, other: Union["Value", float]) -> "Value":
        """
        The addition operation for the Value class.
        Args:
            other (Union["Value", float]): The other value to add to this one.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = x + y
            >>> z.data
            5
        """
        # If the other value is not a Value, then we need to wrap it.
        other = other if isinstance(other, Value) else Value(other)
        # Create a new Value node that will be the output of the addition.
        out = Value(data=self.data + other.data, _children=(self, other), label=f"add")
        def _backward():
            # Local gradient:
            # x = a + b
            # dx/da = 1
            # dx/db = 1
            # Global gradient with chain rule:
            # dy/da = dy/dx . dx/da = dy/dx . 1
            # dy/db = dy/dx . dx/db = dy/dx . 1
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        # Set the backward function on the output node.
        out._backward = _backward
        return out
    def custom_reverse_addition(self, other):
        """
        Reverse addition operation for the Value class.
        Args:
            other (float): The other value to add to this one.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = y + x
            >>> z.data
            5
        """
        # This is the same as adding. We can reuse the __add__ method.
        return self + other
    
    def custom_multiplication(self, other: Union["Value", float]) -> "Value":
        """
        The multiplication operation for the Value class.
        Args:
            other (float): The other value to multiply to this one.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = x * y
            >>> z.data
            6
        """
        # If the other value is not a Value, then we need to wrap it.
        other = other if isinstance(other, Value) else Value(other)
        # Create a new Value node that will be the output of
        # the multiplication.
        out = Value(data=self.data * other.data, _children=(self, other), label=f"multi")
        def _backward():
            # Local gradient:
            # x = a * b
            # dx/da = b
            # dx/db = a
            # Global gradient with chain rule:
            # dy/da = dy/dx . dx/da = dy/dx . b
            # dy/db = dy/dx . dx/db = dy/dx . a
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        # Set the backward function on the output node.
        out._backward = _backward
        return out

    def custom_reverse_multiplication(self, other):
        """
        Reverse multiplication operation for the Value class.
        Args:
            other (float): The other value to multiply to this one.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = y * x
            >>> z.data
            6
        """
        # This is the same as multiplying. We can reuse the __mul__ method.
        return self * other
    
    def custom_power(self, other):
        """
        The power operation for the Value class.
        Args:
            other (float): The other value to raise this one to.
        Usage:
            >>> x = Value(2)
            >>> z = x ** 2.0
            >>> z.data
            4
        """
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        # Create a new Value node that will be the output of the power.
        out = Value(data=self.data ** other, _children=(self,), label=f"power")
        def _backward():
            # Local gradient:
            # x = a ** b
            # dx/da = b * a ** (b - 1)
            # Global gradient:
            # dy/da = dy/dx . dx/da = dy/dx . b * a ** (b - 1)
            self.grad += out.grad * (other * self.data ** (other - 1))
        # Set the backward function on the output node.
        out._backward = _backward
        return out
    
    def custom_negation(self):
        """
        Negation operation for the Value class.
        Usage:
            >>> x = Value(2)
            >>> z = -x
            >>> z.data
            -2
        """
        # This is the same as multiplying by -1. We can reuse the
        # __mul__ method.
        return self * -1
    
    def custom_subtraction(self, other):
        """
        Subtraction operation for the Value class.
        Args:
            other (float): The other value to subtract to this one.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = x - y
            >>> z.data
            -1
        """
        # This is the same as adding the negative of the other value.
        # We can reuse the __add__ and the __neg__ methods.
        return self + (-other)
    def custom_reverse_subtraction(self, other):
        """
        Reverse subtraction operation for the Value class.
        Args:
            other (float): The other value to subtract to this one.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = y - x
            >>> z.data
            1
        """
        # This is the same as subtracting. We can reuse the __sub__ method.
        return other + (-self)
    
    def custom_division(self, other):
        """
        Division operation for the Value class.
        Args:
            other (float): The other value to divide to this one.
        Usage:
            >>> x = Value(10)
            >>> y = Value(5)
            >>> z = x / y
            >>> z.data
            2
        """
        # Use the __pow__ method to implement division.
        return self * other ** -1
    def custom_reverse_division(self, other):
        """
        Reverse division operation for the Value class.
        Args:
            other (float): The other value to divide to this one.
        Usage:
            >>> x = Value(10)
            >>> y = Value(5)
            >>> z = y / x
            >>> z.data
            0.5
        """
        # Use the __pow__ method to implement division.
        return other * self ** -1
    

    # Value.__truediv__ = custom_division
    # Value.__rtruediv__ = custom_reverse_division
    # Value.__sub__ = custom_subtraction
    # Value.__rsub__ = custom_reverse_subtraction
    # Value.__neg__ = custom_negation
    # Value.__pow__ = custom_power
    # Value.__mul__ = custom_multiplication
    # Value.__rmul__ = custom_reverse_multiplication
    # Value.__add__ = custom_addition
    # Value.__radd__ = custom_reverse_addition

    def __repr__(self) -> str:
        """
        Representação em string do objeto Value, mostrando data, rótulo e gradiente.
        """
        return f"Value(data={self.data:.4f}, label='{self.label}', grad={self.grad:.4f})"

    def _ensure_value(self, other) -> 'Value':
        """
        Garante que o 'other' operando seja também um objeto Value.
        Se for um número (int ou float), ele é convertido para Value.
        """
        return other if isinstance(other, Value) else Value(other)

    # --- Operações Aritméticas e Métodos Especiais ---

    def __add__(self, other) -> 'Value':
        """
        Implementa a adição: self + other.
        """
        other = self._ensure_value(other) # Garante que 'other' seja um Value]
        out = Value(self.data + other.data, (self, other), '+', label="Calculation Result")
        
        def _backward():
            # Derivada da soma: d(out)/d(self) = 1, d(out)/d(other) = 1
            # Aplicando a regra da cadeia: dL/d(self) = dL/d(out) * d(out)/d(self)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other) -> 'Value':
        """
        Implementa a multiplicação: self * other.
        """
        other = self._ensure_value(other) # Garante que 'other' seja um Value]
        out = Value(self.data * other.data, (self, other), '*', label="Calculation Result")
        
        def _backward():
            # Derivada do produto: d(out)/d(self) = other.data, d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
          
        return out

    def __pow__(self, other: float) -> 'Value':
        """
        Implementa a potenciação: self ** other (onde 'other' é um escalar int ou float).
        """
        assert isinstance(other, (int, float)), "Apenas potências escalares (int/float) são suportadas por enquanto."
        out = Value(self.data ** other, (self,), f'**{other}', label="Calculation Result")

        def _backward():
            # Derivada de x^n = n * x^(n-1)
            self.grad += (other * (self.data ** (other - 1))) * out.grad #]
        out._backward = _backward

        return out
        
    def __neg__(self) -> 'Value': # -self
        """
        Implementa a negação: -self. É o mesmo que self * -1.
        """
        return self * -1

    def __sub__(self, other) -> 'Value': # self - other
        """
        Implementa a subtração: self - other. É o mesmo que self + (-other).
        """
        return self + (-other)

    def __truediv__(self, other) -> 'Value': # self / other
        """
        Implementa a divisão: self / other. É o mesmo que self * (other ** -1).
        """
        return self * (self._ensure_value(other) ** -1) #]

    # Métodos refletidos para operações onde Value não é o primeiro operando (ex: 2 + a)
    def __radd__(self, other) -> 'Value': # other + self
        return self + other

    def __rsub__(self, other) -> 'Value': # other - self
        return self._ensure_value(other) + (-self)

    def __rmul__(self, other) -> 'Value': # other * self
        return self * other

    def __rtruediv__(self, other) -> 'Value': # other / self
        return self._ensure_value(other) * (self ** -1)

    # --- Funções de Ativação e Outras Funções Matemáticas ---

    def relu(self):
        """
        The ReLU activation function.
        Usage:
            >>> x = Value(-2)
            >>> y = x.relu()
            >>> y.data
            0
        """
        out = Value(data=0 if self.data < 0 else self.data, _children=(self,), label=f"relu")
        def _backward():
            # Local gradient:
            # x = relu(a)
            # dx/da = 0 if a < 0 else 1
            # Global gradient:
            # dy/da = dy/dx . dx/da = dy/dx . (0 if a < 0 else 1)
            self.grad += out.grad * (out.data > 0)
        # Set the backward function on the output node.
        out._backward = _backward
        return out
    def tanh(self) -> 'Value':
        """
        Implementa a função de ativação tangente hiperbólica.
        tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        """
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1) #]
        out = Value(t, (self, ), 'tanh', label=f"tanh")
        
        def _backward():
            # Derivada de tanh(x) = 1 - tanh(x)^2 = 1 - t^2
            self.grad += (1 - t**2) * out.grad #]
        out._backward = _backward
        
        return out

    def exp(self) -> 'Value':
        """
        Implementa a função exponencial: e ** self.data
        """
        x = self.data
        out = Value(math.exp(x), (self,), 'exp', label=f"exp") #]
        
        def _backward():
            # Derivada de e^x = e^x
            self.grad += out.data * out.grad # out.data é e^x]
        out._backward = _backward
        return out

    def log(self, base: float = math.e) -> 'Value':
        """
        Implementa a função logarítmica (log natural por padrão).
        """
        x = self.data
        if x <= 0:
            raise ValueError("Logaritmo indefinido ou complexo para data <= 0.")
        
        # Logaritmo na base desejada
        log_val = math.log(x, base)
        out = Value(log_val, (self,), f'log_base{base:.2f}', label=f"log")
        
        def _backward():
            # Derivada de log_b(x) = 1 / (x * ln(b))
            # Para log natural (ln(x), base=e), a derivada é 1/x.
            self.grad += (1 / (x * math.log(base))) * out.grad
        out._backward = _backward
        return out
    

    # --- Backpropagation ---
    def backward(self):
        """
        Realiza o backpropagation a partir deste Value (geralmente o nó de perda).
        Calcula os gradientes para todos os Values no grafo que levaram a este.

        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = x * y
            >>> z.backward()
            >>> x.grad
            3
            >>> y.grad
            2
        """
        # Build an empty list which will hold the
        # topologically sorted graph
        topo = []
        # Build a set of all the visited nodes
        visited = set()
        # A closure to help build the topologically sorted graph
        def build_topo(node: "Value"):
            if node not in visited:
                # If node is not visited add the node to the
                # visited set.
                visited.add(node)
                # Iterate over the children of the node that
                # is being visited
                for child in node._prev:
                    # Apply recursion to build the topologically sorted
                    # graph of the children
                    build_topo(child)
                
                # Only append node to the topologically sorted list
                # if all its children are visited.
                topo.append(node)
        # Call the `build_topo` method on self
        build_topo(self)
        # Go one node at a time and apply the chain rule
        # to get its gradient
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    