(* Step 1: Generate a 3D random sequence *)
n = 100;
data3D = Table[
   {x, y, Sin[x]*Cos[y] + RandomReal[{-0.5, 0.5}]}, 
   {x, 1, 5, 4/(Sqrt[n])}, {y, 1, 5, 4/(Sqrt[n])}
];
data3D = Flatten[data3D, 1];

(* Step 2: Perform 3D nonlinear fitting with a polynomial model *)
fit3D = NonlinearModelFit[data3D, a*x^2 + b*y^2 + c*x*y + d*x + e*y + f, 
   {a, b, c, d, e, f}, {x, y}];

(* Display the fitting results *)
fit3DParameters = fit3D["BestFitParameters"]
fit3DFunction = fit3D[x, y]

(* Step 3: Plot the original data and the fitted function *)
Show[
 ListPointPlot3D[data3D, PlotStyle -> Red, BoxRatios -> {1, 1, 1}],
 Plot3D[fit3DFunction, {x, 1, 5}, {y, 1, 5}, 
  PlotStyle -> Opacity[0.5, Blue], Mesh -> None]
]

