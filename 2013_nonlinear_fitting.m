
(* use Mathematica : Randomly generate a 100-length sequence, nonlinear fitting  *)

(* Step 1: Generate a 100-length random sequence *)
n = 100;
data = Table[{x, Sin[x] + RandomReal[{-0.5, 0.5}]}, {x, 1, 10, 10/n}];

(* Step 2: Perform nonlinear fitting *)
fit = NonlinearModelFit[data, a*Sin[b*x + c] + d, {a, b, c, d}, x];

(* Display the fitting results *)
fitParameters = fit["BestFitParameters"]
fitFunction = fit["BestFit"]

(* Step 3: Plot the original data and the fitted function *)
Show[
 ListPlot[data, PlotStyle -> Red, PlotLegends -> {"Data"}],
 Plot[fitFunction, {x, 1, 10}, PlotStyle -> Blue, PlotLegends -> {"Fit"}]
]

