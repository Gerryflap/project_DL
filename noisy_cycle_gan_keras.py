
def train_cycle_gan(opts):
    import numpy as np
    # np.random.seed(420)
    # import tensorflow as tf
    # tf.set_random_seed(420)

    n_steps, use_noise, index = opts

    import keras as ks
    import keras.backend as K
    import matplotlib.pyplot as plt

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    def make_trainable(net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    def make_concat_noise(noise_dim):
        def concat_noise(x):
            batch_shape = K.shape(x)[0]
            noise = K.random_normal((batch_shape, noise_dim), 0, 0.25)
            return K.concatenate([x, noise], axis=1)
        return concat_noise


    def build_single_noisy_generator_model(name):

        inp = ks.Input(shape=(2,))
        x = ks.layers.Dense(16, activation='selu')(inp)
        x = ks.layers.Dense(32, activation='selu')(x)
        x = ks.layers.Dense(64, activation='selu')(x)

        x = ks.layers.Lambda(make_concat_noise(64))(x)
        x = ks.layers.Dense(32, activation='selu')(x)
        x = ks.layers.Dense(16, activation='selu')(x)
        out = ks.layers.Dense(2, activation='tanh')(x)

        model = ks.Model(inputs=inp, outputs=out, name=name)
        return model

    def build_single_generator_model(name):

        inp = ks.Input(shape=(2,))
        x = ks.layers.Dense(16, activation='selu')(inp)
        x = ks.layers.Dense(32, activation='selu')(x)
        x = ks.layers.Dense(64, activation='selu')(x)
        x = ks.layers.Dense(32, activation='selu')(x)
        x = ks.layers.Dense(16, activation='selu')(x)
        out = ks.layers.Dense(2, activation='tanh')(x)

        model = ks.Model(inputs=inp, outputs=out, name=name)
        return model


    def build_single_discriminator_model(name):
        model = ks.models.Sequential(name=name)
        model.add(ks.layers.Dense(64, activation='relu', input_shape=(2,)))
        model.add(ks.layers.Dense(32, activation='relu'))
        model.add(ks.layers.Dense(16, activation='relu'))
        model.add(ks.layers.Dense(1, activation='sigmoid'))
        return model


    def build_generator_loss(discriminator):
        def generator_loss(y_pred, y_target):
            return ks.losses.binary_crossentropy(discriminator(y_pred), y_target)
        return generator_loss


    def sample_random(array, size):
        return array[np.random.randint(0, array.shape[0], size)]


    X_data = np.concatenate([np.random.normal(-0.5, 0.1, (10000, 1)), np.random.normal(0.5, 0.1, (10000, 1))], axis=1)
    Y_data = np.concatenate([np.random.normal(0.5, 0.1, (10000, 1)), np.random.normal(0.5, 0.1, (10000, 1))], axis=1)

    # plt.scatter(X_data[:, 0], X_data[:, 1])
    # plt.scatter(Y_data[:, 0], Y_data[:, 1])
    # plt.show()

    if use_noise:
        G_X_Y = build_single_noisy_generator_model("G_xy")
        G_Y_X = build_single_noisy_generator_model("G_yx")
    else:
        G_X_Y = build_single_generator_model("G_xy")
        G_Y_X = build_single_generator_model("G_yx")
    D_X = build_single_discriminator_model("D_x")
    D_Y = build_single_discriminator_model("D_y")

    D_X.compile(ks.optimizers.Adam(0.0003), loss=ks.losses.binary_crossentropy)
    D_Y.compile(ks.optimizers.Adam(0.0003), loss=ks.losses.binary_crossentropy)

    # Define X -> Y -> X cycle model
    X_input = ks.Input(shape=(2,))
    generated_Y = G_X_Y(X_input)
    make_trainable(D_Y, False)
    discriminator_Y_pred = D_Y(generated_Y)
    X_cycle_out = G_Y_X(generated_Y)

    X_cycle_model = ks.Model(inputs=X_input, outputs=[discriminator_Y_pred, X_cycle_out])
    X_cycle_model.compile(ks.optimizers.Adam(0.0003), loss=[ks.losses.binary_crossentropy, ks.losses.mean_squared_error])

    # Define Y -> X -> Y cycle model
    Y_input = ks.Input(shape=(2,))
    generated_X = G_Y_X(Y_input)
    make_trainable(D_X, False)
    discriminator_X_pred = D_X(generated_X)
    Y_cycle_out = G_X_Y(generated_X)

    Y_cycle_model = ks.Model(inputs=Y_input, outputs=[discriminator_X_pred, Y_cycle_out])
    Y_cycle_model.compile(ks.optimizers.Adam(0.0003), loss=[ks.losses.binary_crossentropy, ks.losses.mean_squared_error])

    # Training Loop:
    epochs = n_steps
    batch_size = 32

    G_batches_per_epoch = 4
    D_batches_per_epoch = 1

    real_X_fixed = sample_random(X_data, 5)
    real_Y_fixed = sample_random(Y_data, 5)

    for i in range(epochs):

        # Update Discriminators
        for j in range(D_batches_per_epoch):
            real_X = sample_random(X_data, batch_size//2)
            real_Y = sample_random(Y_data, batch_size//2)

            fake_X = G_Y_X.predict(real_Y)
            fake_Y = G_X_Y.predict(real_X)

            D_d_labels = np.zeros((batch_size, 1))
            D_d_labels[batch_size//2:] = 1

            D_g_labels = np.zeros((batch_size, 1))
            D_g_labels[batch_size//2:] = 1

            D_d_inputs = np.concatenate([fake_X, real_X], axis=0)
            D_g_inputs = np.concatenate([fake_Y, real_Y], axis=0)

            h = D_X.fit(D_d_inputs, D_d_labels, batch_size=batch_size, verbose=False)
            D_X_loss = h.history['loss'][0]
            h = D_Y.fit(D_g_inputs, D_g_labels, batch_size=batch_size, verbose=False)
            D_Y_loss = h.history['loss'][0]


        # Update Generators
        for j in range(G_batches_per_epoch):
            real_X = sample_random(X_data, batch_size)
            real_Y = sample_random(Y_data, batch_size)

            labels = np.ones((batch_size, 1))

            h = X_cycle_model.fit(real_X, [labels, real_X], batch_size=batch_size, verbose=False)
            X_cycle_loss = h.history['G_yx_loss'][0]
            Y_gen_loss = h.history['D_y_loss'][0]
            h = Y_cycle_model.fit(real_Y, [labels, real_Y], batch_size=batch_size, verbose=False)
            Y_cycle_loss = h.history['G_xy_loss'][0]
            X_gen_loss = h.history['D_x_loss'][0]

        # Show results
        if i%100 == 0:

            plt.scatter(X_data[:, 0], X_data[:, 1])
            plt.scatter(Y_data[:, 0], Y_data[:, 1])
            plt.scatter(real_X_fixed[:, 0], real_X_fixed[:, 1])
            for _ in range(6):
                predicted_Y = G_X_Y.predict(real_X_fixed)
                plt.scatter(predicted_Y[:, 0], predicted_Y[:, 1])
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.savefig("simple_cycle_gan/X_Y_%d_%s_%d.png" % (index, "noisy" if use_noise else "plain", i))
            plt.clf()


            print("Episode %d:"%i)
            print("Discriminator losses: X = %.3f \t Y = %.3f"%(D_X_loss, D_Y_loss))
            print("D->G->D losses: Gen = %.3f \t Cycle = %.3f"%(Y_gen_loss, X_cycle_loss))
            print("G->D->G losses: Gen = %.3f \t Cycle = %.3f"%(X_gen_loss, Y_cycle_loss))
            print()




steps = 10000

import multiprocessing as mp

pool = mp.Pool(processes=8)

experiments = [(steps, use_noise, i) for use_noise in [True, False] for i in range(4)]
# experiments = [(steps, use_noise, i) for use_noise in [False] for i in range(8)]


x = pool.imap_unordered(train_cycle_gan, experiments)
for x_v in x:
    print(x_v)
