
# Setup

To reproduce the experimental results, first clone the Marabou branch below and install the tool:

```
git clone https://github.com/anwu1219/Marabou
git checkout softmax-bound
mkdir build
cd build
cmake ../
make -j12
```


# Models

the `training` folder contains several transformer models trained on the MNIST and SST datasets.

The transformer model used in Table 4 are self-attention-mnist-pgd-{small,medium,large}-sim.onnx

The transformer model used in Table 5 is self-attention-sst-sim.onnx

# Running the experiment

One can use `run_experiment_{mnist,sst}.sh` to run robustness verification on the models:

```run_experiment_{mnist,sst}.sh [network] [index] [epsilon] [bound type]```

The softmax bound type is one of linear, er, lse1, lse2

For example:

`run_experiment_mnist.sh self-attention-mnist-pgd-big-sim.onnx 499 0.008 lse1`

This should print "unsat" in the last line.

In general, the result is either *unknown*, which means the network is not necessarily robust, or *unsat*, meaning the network is robust w.r.t. the test image and the perturbation bound.

To reproduce Table 4, run `run_experiment_mnist.sh` for each line in all_arguments_mnist

To reproduce Table 5, run `run_experiment_sst.sh` for each line in all_arguments_sst
