# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Non-linear operators
"""

from cntk.ops.cntk1 import Abs, Exp, RectifiedLinear, Sigmoid, Softmax, Tanh

def rectified_linear(x, name=None):
    """
    computes the element-wise rectified linear of `x`: ``max(x, 0)``

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`

    Example:
        >>> cntk.eval(cntk.ops.rectified_linear([[-1, -0.5, 0, 1, 2]]))
        [[[0, 0, 0, 1, 2]]]
    """
    return RectifiedLinear(x, var_name=name)

def sigmoid(x, name=None):
    """
    computes the element-wise sigmoid of `x`: 

    :math:`sigmoid(x) = {1 \over {1+\exp(-x)}}`

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Sigmoid(x, var_name=name)

def tanh(x, name=None):
    """
    computes the element-wise tanh of `x`: 

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Tanh(x, var_name=name)

def softmax(X, name=None):
    """
    computes the element-wise sigmoid of `X`: 

    :math:`softmax(x) = {\exp(x) - \max_{x \in X}(\exp(x)) \over {\sum_{x \in
    X} \exp(x)- \max_{x \in X}(\exp(x)) }}`

    The term :math:`\max_{x \in X}(\exp(x))` is subtracted for numerical
    stability.

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`

    Examples:
        >>> cntk.eval(cntk.ops.softmax([[1, 1, 2, 3]]))
        [[[0.08259454, 0.08259454, 0.22451524, 0.61029569]

        >>> cntk.eval(cntk.ops.softmax([[1, 1]]))
        [[[0.5, 0.5]]]
    """
    return Softmax(X)

def exp(x, name=None):
    """
    computes the element-wise exponential of `x`: 

    :math:`exp(x) = {e^x}`

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Exp(x, var_name=name)

def abs(x, name=None):
    """
    computes the element-wise absolute of `x`: 

    :math:`abs(x) = |x|`

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Abs(x, var_name=name)
