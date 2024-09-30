(* 1. Generate synthetic data *)
data = Table[{x, 2 x + 1 + RandomReal[{-0.2, 0.2}]}, {x, -1, 1, 0.02}];
ListPlot[data, PlotStyle -> Blue, PlotRange -> All, PlotLabel -> "Synthetic Data"]

(* 2. Define the linear model and loss function *)
model[m_, b_][x_] := m x + b
lossFunction[m_, b_] := Total[(model[m, b][#[[1]]] - #[[2]])^2 & /@ data]

(* 3. Perform the optimization to minimize the loss *)
{minLoss, {mOpt, bOpt}} = FindMinimum[lossFunction[m, b], {{m, 0}, {b, 0}}]

(* 4. Plot the results *)
fittedModel = model[mOpt, bOpt];
Show[
    ListPlot[data, PlotStyle -> Blue, PlotLabel -> "Model Fit vs Actual Data"],
    Plot[fittedModel[x], {x, -1, 1}, PlotStyle -> Red]
]

(* Optionally, visualize the loss function *)
Plot3D[lossFunction[m, b], {m, mOpt - 0.5, mOpt + 0.5}, {b, bOpt - 0.5, bOpt + 0.5},
    PlotRange -> All, PlotLabel -> "Loss Function", AxesLabel -> {"m", "b", "Loss"}]

