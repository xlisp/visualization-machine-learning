(* 1. Define Predicted Values as Variables *)
yPred = Array[Subscript[yPred, #] &, 10];  (* Creates {yPred_1, yPred_2, ..., yPred_10} *)

(* Define True Values *)
yTrue = {2.5, 0.0, 2.0, 8.0, 4.5, 6.0, 1.5, 3.5, 7.0, 5.0};

(* 2. Compute the Loss Function *)
mseLoss = Mean[(yPred - yTrue)^2];

(* 3. Compute the Gradients (Derivatives) *)
gradients = D[mseLoss, #] & /@ yPred;

(* Display the Gradients *)
gradients

