import numpy as np
import pytato as pt

n = 5
q = pt.make_placeholder(name="state", shape=(n,), dtype=np.float64)
mat = pt.make_placeholder(name="mat", shape=(n, n), dtype=np.float64)

def flux_func(qi, qj):
    return 0.5*(qi + qj)

# Need something like: (shout out to Kaushik for this trick)
# qi = actx.np.expand_dims(q, axis=-1)
# qj = actx.np.expand_dims(q, axis=-2)
# for data coming from DOFArrays
out = mat @ flux_func(q.reshape(-1, 1), q.reshape(1, -1))

result = pt.DictOfNamedArrays({"out": out})

# {{{ generate OpenCL code

import loopy as lp

prg = pt.generate_loopy(result)

print(lp.generate_code_v2(lp.simplify_indices(prg.program)).device_code())

# }}}
