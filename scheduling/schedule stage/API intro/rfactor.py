import tvm

n = 1024
k = tvm.reduce_axis((0, n), name='k')

A = tvm.placeholder((n,), name='A')
B = tvm.compute((1,), lambda i: tvm.sum(A[k], axis=k), name='B')

s = tvm.create_schedule(B.op)
ko, ki = s[B].split(s[B].op.reduce_axis[0], 32)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

BR = s.rfactor(B, ki)

print(tvm.lower(s, [A, B], simple_mode=True))