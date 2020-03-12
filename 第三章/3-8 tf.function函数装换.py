import tensorflow as tf
from tensorflow import keras

def scaled_elu(z,scale=1.0,alpha=1.0):
    #z>=0?scale*z:scale*alpha*tf.nn.elu(z)
    is_positive = tf.greater_equal(z,0.0)
    return scale*tf.where(is_positive,z,alpha*tf.nn.elu(z))

@tf.function
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
    display(Markdown('```python\n{}\n```'.format(code)))


display_tf_code(scaled_elu)
display_tf_code(converge_to_2)


var = tf.Variable(0.)
@tf.function
def add_21():
    return var.assign_add(21)

print(add_21())


@tf.function
def cube(z):
    return tf.pow(z,3)
print(cube(tf.constant([11.,2.,3.])))
print(cube(tf.constant([1,2,3])))
