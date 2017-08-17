# 1D_GAN
Simple tensorflow implementation of 1D convolutional Generative Adversarial Network.

## Run instructions
Assuming you installed Python 3 with *tensorflow*, *numpy* and *pandas* you need to:  

- Clone the project
```bash
git clone https://github.com/PaulEmmanuelSotir/1D_GAN.git
cd ./1D_GAN
```

- And train the model
```bash
# By default, the model will be trained on a 2 channels timeserie of bitcoin to eur exchange rate and volumes (csv file in *./data* directory)
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
floyd run --data paulemmanuel/datasets/btc_eur_1y/1 --env tensorflow-1.2 --tensorboard --gpu "python gan1d.py --floyd-job"
```
