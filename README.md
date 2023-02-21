# Adaptive noise cancellation with a neural network

Implemented with Tensorflow. Parameters can be adjusted at the top of the script. The inputs of the neural network is the noised sound signal from the current to n steps ago with n being the number of tapped delays. The target of the neural network is the original sound signal at the next step.

![](test)
