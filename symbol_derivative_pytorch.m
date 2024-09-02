#!/Applications/Mathematica.app/Contents/MacOS/WolframScript -script

(* Implement this algorithm using  Mathematica => Define the neural network parameters *)
l = 3; (* Number of layers *)
inputDim = 10;
outputDim = 1;
hiddenLayers = {5, 5}; (* Sizes of hidden layers *)

(* Initialize weights and biases *)
W[i_] := RandomReal[{-1, 1}, {hiddenLayers[[i]], If[i == 1, inputDim, hiddenLayers[[i - 1]]]}]
b[i_] := RandomReal[{-1, 1}, hiddenLayers[[i]]]

(* Define the activation function, assuming it's ReLU *)
f[a_] := Max[0, a]

(* Forward propagation *)
h[0] = x; (* Input *)
a[k_] := b[k] + W[k].h[k - 1]
h[k_] := f[a[k]]

(* Output layer *)
W[l] = RandomReal[{-1, 1}, {outputDim, hiddenLayers[[-1]]}];
b[l] = RandomReal[{-1, 1}, outputDim];
yHat = W[l].h[l - 1] + b[l];

(* Define the loss function *)
L[yHat_, y_] := Norm[yHat - y]^2
Omega[theta_] := Norm[theta]^2 (* Regularization term *)

J = L[yHat, y] + Omega[theta];

(* Display the result *)
print["--------result--------"]
print[J]

