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
from .ops_test_utils import unittest_helper, C, AA, I, precision
from ...graph import *
from ...reader import *
from ..non_linear import if_then_else
import numpy as np

IF_THEN_ELSE_TUPLES = [ 
						([-1], [2], [3]), 
                        ([0], [20], [30]),
						([10],[0],[-100])
					  ]
					  
# -- if_then_else operation tests --
@pytest.mark.parametrize("cond, value_a, value_b", IF_THEN_ELSE_TUPLES)
def test_op_if_then_else(cond, value_a, value_b, device_id, precision):    

    #Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # Compare to numpy's implementation of where()
    expected = [[np.where(AA(cond), AA(value_a), AA(value_b))]]

    a = I([cond], has_sequence_dimension=False)
    b = C([value_a])    
    c = C([value_b])
    
    result = if_then_else(a, b, c)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)