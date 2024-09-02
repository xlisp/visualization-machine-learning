#!/Applications/Mathematica.app/Contents/MacOS/WolframScript -script

## rewrite least_squares_method.py it to Mathematica

(* Example data points *)
X = {1, 2.2, 3, 4, 5.2};
y = {2, 4.1, 6.3, 8, 10.1};

(* Add a column of ones to X for the intercept term *)
Xb = Transpose[{ConstantArray[1, Length[X]], X}];

(* Calculate the best fit line parameters using the Normal Equation *)
thetaBest = Inverse[Transpose[Xb].Xb].Transpose[Xb].y

(* Extract the parameters (intercept and slope) *)
intercept = thetaBest[[1]];
slope = thetaBest[[2]];

(* Predict the y values for the best fit line *)
yPred = intercept + slope*X;

(* Plot the data points and the best fit line *)
ListPlot[
    {
        Transpose[{X, y}], (* Data points *)
        Table[{x, intercept + slope*x}, {x, Min[X], Max[X], 0.1}] (* Best fit line *)
    },
    PlotStyle -> {Blue, Red},
    PlotLegends -> {"Data points", "Best fit line"},
    AxesLabel -> {"X", "y"}
]

