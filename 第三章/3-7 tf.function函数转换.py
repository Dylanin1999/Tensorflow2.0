import tensorflow as tf
from tensorflow import keras

#tf.function and autograph
def scaled_elu(z,scale=1.0,alpha=1.0):
    #z>=0?scale*z:scale*alpha*tf.nn.elu(z)
    is_positive = tf.greater_equal(z,0.0)
    return scale*tf.where(is_positive,z,alpha*tf.nn.elu(z))

print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3,-2.5])))
scaled_elu_tf = tf.function(scaled_elu())
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3., -2.5])))
print(scaled_elu_tf.python_function is scaled_elu_tf)

#%timeit scaled_elu(tf.random.normal((1000,1000)))
#%timeit scaled_elu(tf.random.normal((1000,1000)))
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2.0
    return total
print(converge_to_2(20))

def display_tf_code(func):
    code = tf.autograph.to_code(func)
    from IPython.display import display, Markdown
    display(Markdown('''python\n{}\n'''.format()))
