import tensorflow as tf


def compute_minimum_of_expression(learning_rate):
    # Avoid polluting the default graph by using an alternate graph
    with tf.Graph().as_default():
        tf.set_random_seed(1234)

        # Create two scalar variables, x and y, initialized at random.
        x = tf.get_variable(name='x', shape=[], dtype=tf.float32, initializer=tf.random_normal_initializer)
        y = tf.get_variable(name='y', shape=[], dtype=tf.float32, initializer=tf.random_normal_initializer)

        # Create a tensor z whose value represents the expression
        #     2(x - 2)^2 + 2(y + 3)^2
        z = 2 * (x - 2) ** 2 + 2 * (y + 3) ** 2

        # Compute the gradients of z with respect to x and y.
        dx, dy = tf.gradients(ys=z, xs=[x, y])

        # Create an assignment expression for x using the update rule
        #    x <- x - 0.1 * dz/dx
        # and do the same for y.
        x_update = tf.assign_sub(x, learning_rate * dx)
        y_update = tf.assign_sub(y, learning_rate * dy)

        with tf.Session() as session:
            # Run the global initializer op for x and y.
            session.run(tf.global_variables_initializer())

            print("--------------- Computing minimum of expression with manual gradient computation ---------------\n")

            for _ in range(10):
                # Run the update ops for x and y.
                session.run([x_update, y_update])

                # Retrieve the values for x, y, and z, and print them.
                x_val, y_val, z_val = session.run([x, y, z])
                print('x = {:4.2f}, y = {:4.2f}, z = {:4.2f}'.format(x_val, y_val, z_val))


def compute_minimum_of_expression_with_tf_sgd(learning_rate):
    with tf.Graph().as_default():
        tf.set_random_seed(1234)

        # Create two scalar variables, x and y, initialized at random.
        x = tf.get_variable(name='x', shape=[], dtype=tf.float32, initializer=tf.random_normal_initializer)
        y = tf.get_variable(name='y', shape=[], dtype=tf.float32, initializer=tf.random_normal_initializer)

        # Create a tensor z whose value represents the expression
        #     2(x - 2)^2 + 2(y + 3)^2
        z = 2 * (x - 2) ** 2 + 2 * (y + 3) ** 2

        # Create a gradient descent optimizer with a specific learning rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        update_op = optimizer.minimize(loss=z, var_list=tf.trainable_variables())

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            print("\n--------- Computing minimum of expression with Tensorflow gradient descent optimizer ---------\n")

            for _ in range(10):
                # Run the update ops for x and y.
                session.run(update_op)

                # Retrieve the values for x, y, and z, and print them.
                x_val, y_val, z_val = session.run([x, y, z])
                print('x = {:4.2f}, y = {:4.2f}, z = {:4.2f}'.format(x_val, y_val, z_val))


if __name__ == "__main__":
    LEARNING_RATE = 0.1

    compute_minimum_of_expression(LEARNING_RATE)
    compute_minimum_of_expression_with_tf_sgd(LEARNING_RATE)
