import tensorflow as tf
impolrt tensorflow.layers as layers

def mlp_model(inputs, num_outputs, scope, reuse=False, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_nuits, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_nuits, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def critic(scope, q_func, reuse=None):
    with tf.variable_scope(scope, reuse=reuse)
        


def actor(scope, q_func, p_func, reuse=None):
    with tf.variable_scope(scope, reuse=reuse)
        ssZZ



class ActorCritic:
    def __init__(self, name, model):
        self.critic = cirtic(scope=name, q_func=model)
        self.actor = actor(scope=name, q_func=model, p_func=model)


if __name__ == "__main__":
    ac = ActorCritic(name="ac", model=mlp_model)
    with tf.Session() as sess:
       print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    
