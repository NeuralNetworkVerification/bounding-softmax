
- robust_mnist2x10.onnx

  A pgd-trained mnist classifier

- save_weights.py

  Save the weights and biases as a .npz file

- robust_mnist2x10.npz

  Result of running save_weights.py

- print_weights.py

  Example of loading the .npz file, and evaluating the network.
  The result matches bounds/bounds_ind0_eps0.0

- bounds/

  The bounds folder contains the bounds of neurons given an l-inf ball with radius epsilon around an image derived by CROWN.
