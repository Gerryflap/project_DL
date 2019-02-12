# project_DL
Code for the GAN project. A large part of this code has been provided to us and is not ours.

## How to run experiments:

### Generative CycleGAN
Run `gan_die_all_kan_test.py` with parameters similar to the `cycle_gan.py`. 
The noise dimension can be edited in the model declaration in `gan_die_alles_kan.py`

Additionally the 2D code can be run by running `noisy_cycle_gan_keras.py`. This does require a working Keras installation.

### Replay GAN
Replay GAN is in a separate branch (sorry). It can be executed by running `replay_gan.py` similar to `vanilla_gan.py`. 
Extra console parameters are added to control the extra hyperparameters.

### Threshold GAN
The Threshold GAN is in the master branch and can be run as `threshold_gan.py` using normal parameters. 
One can use the extra console parameter `--gen_loss_threshold` to control the maximum generator loss threshold.
