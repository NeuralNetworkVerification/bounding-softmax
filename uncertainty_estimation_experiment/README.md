
# Ensemble models

the networks folder contains several ensemble models trained on the MNIST and CIFAR10 datasets. Each individual model is stored in onnx format and the weights are also stored in a correspoding npz file.

The ensemble model used in Table 1 of the paper is mnist-{0,1,2,3,4}.onnx.

The ensemble model used in Table 2 is robust_cifar10-{0,1,2,3,4}.onnx.

The ensemble model used in Table 10 is robust_mnist-large-{0,1,2,3,4}.onnx.

# bounds

The bound folder contains the neuron bounds of a network on l-inf eps-ball around a mnist/cifar10 test image. For example, `bounds_net0_ind0_eps0.008` contains the neuron bounds of `mnist-0.onnx` given a perturbation of radius 0.008 around the first test image.

These bounds are generated using the DeepPoly analysis in Marabou.

# Running the experiment

Install the python packages in `requirements.txt`

The main experiment script is `UQ_Verif_Deep_Ensemble_cluster.py`, which performs the uncertainty quantification tasks as described in Sec. 7.1. The script requires 6 arguments:

- --network: the network to verify
- --eps: the l-inf perturbation on the input
- --index: the index of the test image
- --lb: types of softmax lower-bound to use
- --ub: types of softmax upper-bound to use
- --scoring: the scoring function

Run `UQ_Verif_Deep_Ensemble_cluster.py --help` to see the possible values for these arguments.

For example:

`python UQ_Verif_Deep_Ensemble.py --network mnist --eps 0.008 --index0 --lb LSE --ub LSE --scoring Brier`

If the experiment is successfully run, an pickle file `mnist_ind0_eps0.008_lbLSE_ubLSE_scoreBrier_results.pickle` will be created in the result folder.

All the input arguments required to reproduce Tables 1, 2, and 10 are contained in `all_arguments`

To get the average uncertainty quantification score of a given network w.r.t. a score metric (e.g., Brier or NLL), run:

`python read_results.py Brier mnist`

This will print a 3 by 3 matrix, where the rows correspond to bound type: lin, ER, LSE,
and the columns correspond to perturbation levels 0.008, 0.012, 0.016. The entry of the table is the average uncertainty quantification score over all the instances runned so far.

To run the analysis mode described in in Appendix I.3, use `UQ_Verif_Deep_Ensemble_separate.py` and `read_results_separate.py" instead.