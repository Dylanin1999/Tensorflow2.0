import tensorflow as tf
from tensorflow import keras

@tf.function(input_signature=[tf.TensorSpec([None],tf.int32,name='x')])
def cube(z):
    return tf.pow(z,3)

try:
    print(cube(tf.constant([1.,2.,3.])))
except ValueError as ex:
    print(ex)

print(cube(tf.constant([1,2,3])))

#@tf.function py func -> tf graph
#get_concrete_function -> add input signature -> SaveModel

cube_func_int32 = cube.get_concrete_function(
    tf.TensorSpec([None],tf.int32)
)
print(cube_func_int32)

print(cube_func_int32 is cube.get_concrete_function(tf.TensorSpec([5],tf.int32)))
print(cube_func_int32 is cube.get_concrete_function(tf.constant([1,2,3])))

cube_func_int32.graph
cube_func_int32.graph.get_operations()
pow_op = cube_func_int32.graph.get_operations()[2]
print(pow_op)

print(list(pow_op.inputs))
print(list(pow_op.outputs))

cube_func_int32.graph.get_operation_by_name("x")

cube_func_int32.graph.get_operation_by_name("x:0")

cube_func_int32.graph.as_graph_def()

