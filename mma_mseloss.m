(* 1. Define Data *)
yTrue = {2.5, 0.0, 2.0, 8.0, 4.5, 6.0, 1.5, 3.5, 7.0, 5.0};  (* target values *)
yPred = {3.0, -0.5, 2.0, 7.0, 5.5, 5.0, 2.5, 3.0, 8.0, 4.0};  (* predicted values *)

(* 2. Compute Mean Squared Error *)
squaredErrors = (yPred - yTrue)^2;
mse = Mean[squaredErrors];

(* 3. Compute Gradients (for each prediction) *)
gradients = 2*(yPred - yTrue)/Length[yTrue];

(* 4. Plot the Results *)
(* Plot Squared Errors *)
squaredErrorsPlot = BarChart[squaredErrors, 
  ChartLabels -> Range[Length[squaredErrors]], 
  ChartStyle -> Orange, 
  PlotLabel -> "Squared Errors", 
  AxesLabel -> {"Data Points", "Squared Error"},
  GridLines -> {None, {mse}},
  GridLinesStyle -> Directive[Red, Dashed]
];

(* Plot Gradients *)
gradientsPlot = BarChart[gradients, 
  ChartLabels -> Range[Length[gradients]], 
  ChartStyle -> Blue, 
  PlotLabel -> "Gradients of Predicted Values", 
  AxesLabel -> {"Data Points", "Gradient"}
];

(* Display the Plots Together *)
GraphicsGrid[{{squaredErrorsPlot}, {gradientsPlot}}]

