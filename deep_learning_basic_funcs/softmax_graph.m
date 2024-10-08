
(* graph show the  pytorch softmax => how to use Mathematica express it ?  \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} *)

softmax[x_List] := Exp[x]/Total[Exp[x]]


input = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0};
output = softmax[input]


BarChart[output, ChartLabels -> input, 
 ChartStyle -> Blue, 
 PlotLabel -> "Softmax Function", 
 AxesLabel -> {"Input", "Softmax Output"}]


(* plot  softmax function *)

Plot[Evaluate[softmax[{x, 2, 3, 4}]], {x, -5, 5},
 PlotLegends -> {"softmax[{x, 2, 3, 4}]"},
 PlotRange -> All, AxesLabel -> {"x", "softmax output"}, 
 PlotLabel -> "Softmax Function"]


Plot3D[Evaluate[softmax[{x, y, 2}][[1]]], {x, -5, 5}, {y, -5, 5},
 PlotRange -> All, AxesLabel -> {"x", "y", "softmax output"}, 
 PlotLabel -> "Softmax Function"]

