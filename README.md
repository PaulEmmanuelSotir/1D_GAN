# 1D_GAN (WIP)

Tensorflow implementation of 1D convolutional Generative Adversarial Network (improved WGAN variant, see [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)).

### Be aware this is a Work In Progress (see https://github.com/PaulEmmanuelSotir/1D_GAN/issues/1)

## Run instructions

Assuming you installed Python 3 with *tensorflow*, *numpy* and *pandas* you need to:

- Clone the project

```bash
git clone https://github.com/PaulEmmanuelSotir/1D_GAN.git
cd ./1D_GAN
```

- And train the model

```bash
# By default, the model will be trained on sinusoidal curves of random frequency and offset
python gan1d.py
```

You can see training progress on tensorboard:

```bash
# Launch the following command and browse to localhost:6006
tensorboard --logdir=./models
```

Also note that this project can run on [Floyd](https://www.floydhub.com/) (Heroku for deep learning):

```bash
# To run a Floyd training job, use the following command:
floyd run --data paulemmanuel/datasets/btc_eur_1y/1 --env tensorflow-1.4 --tensorboard --gpu "python gan1d.py --floyd-job"
```
