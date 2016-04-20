# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================


from cntk.ops.cntk1 import If

def cond(flag, value_if_true, value_if_false, name=None):
    """
    Return either value_if_true or value_if_false based on the value of flag.
    If flag != 0 value_if_true is returned, otherwise value_if_false.
    
    Args:
        flag: tensor
        value_if_true: tensor
        value_if_false: tensor            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return If(flag, value_if_true, value_if_false, var_name = name)