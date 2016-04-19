# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================


from cntk.ops.cntk1 import If

def if_then_else(condition, value1, value2, name=None):
    """
    Based on first tensor value select one of the two other values. 
    For the input tensor x, this node outputs a tensor of the same shape with 
    all of its values clipped to fall between min_value and max_value.
    The backward pass propagates the received gradient if no clipping occurred,
    and 0 if the value was clipped.
    
    Args:
        x: tensor to be clipped
        condition: the minimum value to clip element values to
        value1: the maximum value to clip element values to
        value2: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return If(condition, value1, value2, var_name = name)