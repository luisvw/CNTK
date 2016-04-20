# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================

"""
Unit tests for non-linear operations. Each operation is tested for 
the forward and the backward pass
"""

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, C, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *
from ..non_linear import if_then_else
import numpy as np

IF_THEN_ELSE_TUPLES = [ 
						([-1], [2], [3]), 
                        ([0], [20], [30]),
						([10],[0],[-100]),
					  ]
					  
# -- if_then_else operation tests --
@pytest.mark.parametrize("cond, value_a, value_b", IF_THEN_ELSE_TUPLES)
def test_op_if_then_else(cond, value_a, value_b, device_id, precision):    

    #Forward pass test
    #==================
    # Comparing to numpy's implementation of where(...)

    expected = [[[np.where(AA(cond, dtype=PRECISION_TO_TYPE[precision]), AA(value_a, dtype=PRECISION_TO_TYPE[precision]), AA(value_b, dtype=PRECISION_TO_TYPE[precision]))]]]

    cond_as_const    = C([cond])
    value_a_as_const = C([value_a])    
    value_b_as_const = C([value_b])   
	
    cond_as_input    = I([cond],    has_sequence_dimension=False)
    value_a_as_input = I([value_a], has_sequence_dimension=False)
    value_b_as_input = I([value_b], has_sequence_dimension=False)
	
    result = if_then_else(cond_as_input, value_a_as_const, value_b_as_const)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
					
	#Backward pass test
    #==================
    # The derivative of the if_then_else() function is zero for the first argument.
	# The derivative for second and thrird argument depends on the first:
	# * Derivative of second argument = derivative of input if cond else 0
	# * Derivative of third argument  = derivative of input if not cond else 0

    # Derivative for first parameter should always be zero
    expected  = [[[np.zeros_like(x) for x in cond]]]
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=cond_as_input)

    # Derivative of second parameter depends on cond
    expected = [[np.array(np.where(cond, 1, 0), dtype=PRECISION_TO_TYPE[precision])]]
    result = if_then_else(cond_as_const, value_a_as_input, value_b_as_const)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=value_a_as_input)

    # Derivative of third parameter depends on cond
    expected = [[np.array(np.where(cond, 0, 1), dtype=PRECISION_TO_TYPE[precision])]]
    result = if_then_else(cond_as_const, value_a_as_const, value_b_as_input)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=value_b_as_input)