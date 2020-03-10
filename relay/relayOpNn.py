from tvm import relay
import tvm
# The following 4 lines are equivalent to each other
x = tvm.relay.Var("x", tvm.relay.TensorType([1, 2]))
x = tvm.relay.var("x", tvm.relay.TensorType([1, 2]))
x = tvm.relay.var("x", shape=[1, 2])
x = tvm.relay.var("x", shape=[1, 2], dtype="float32")

# The following 2 lines are equivalent to each other.
y = tvm.relay.var("x", "float32")
y = tvm.relay.var("x", shape=(), dtype="float32")
print(x)
print(y)
relay.nn.dense()