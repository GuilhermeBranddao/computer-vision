from learn.toolkit.engine import Value
from learn.toolkit.nn import MLP
from typing import List, Union


# Supondo que você tenha a função draw_dot como nos notebooks micrograd
# Se precisar, aqui está uma versão simplificada (requer 'graphviz' instalado: pip install graphviz)
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    from graphviz import Digraph
    """
    format: png | pdf | svg
    rankdir: TB (top-bottom graph) | LR (left-right graph)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': rankdir})

    for n in nodes:
        label = f"data {n.data:.4f} | grad {n.grad:.4f} | label {n.label}"
        dot.node(name=str(id(n)), label = label, shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

# Função genérica para converter qualquer vetor/lista y em lista de objetos Value (com rótulo opcional)
def to_value_vector(y, label_prefix="y"):
    """
    Converte um vetor/lista y em uma lista de objetos Value, com rótulo customizado.
    """
    return [Value(v, label=f"{label_prefix} {i}") for i, v in enumerate(y)]

def to_value_matrix(X, label_prefix="input"):
    """
    Converte uma matriz X (list of lists ou np.ndarray) em uma lista de listas de objetos Value,
    com rótulo customizado para cada entrada.
    """
    return [
        [Value(v, label=f"{label_prefix} {i}_{j}") for j, v in enumerate(row)]
        for i, row in enumerate(X)
    ]


def forward(mlp: "MLP", x_data:List[List[float]]) -> List["Value"]:
    """
    The forward function takes the mlp and the inputs. The inputs are forwarded through the mlp, and we obtain the predictions from the mlp.
    """
    # Get the predictions upon forwarding the input data through the mlp
    y_pred_list = [mlp(x) for x in x_data]
    return y_pred_list

def update_mlp(mlp: "MLP", learning_rate:float):
    """
    In this function, we update the parameters (weights and biases) of our mlp with the gradient information.

    **Como funciona o ajuste?**
    - Para cada parâmetro p no mlp:
        - Se p.grad é positivo (aumentar p aumenta a perda), então p.data é diminuído (pois subtraímos `learning_rate` * p.grad, que é um valor positivo).
        - Se p.grad é negativo (aumentar p diminui a perda), então p.data é aumentado (pois subtraímos `learning_rate` * p.grad, que é um valor negativo, resultando em uma adição).
    
    - O `learning_rate` (taxa de aprendizado) controla o "tamanho do passo" que damos ao ajustar os parâmetros. 
        - Uma taxa muito alta pode fazer o modelo "pular" o ponto ótimo; 
        - uma taxa muito baixa pode tornar o aprendizado excessivamente lento.
    - O resultado líquido é que todos os pesos e biases do mlp são sutilmente alterados em uma direção que, espera-se, reduza a total_loss na próxima vez que o forward pass for executado com os mesmos dados.
    """
    # Update (Gradiente Descendente)
    for layer in mlp.parameters():
        layer.data -= learning_rate * layer.grad
