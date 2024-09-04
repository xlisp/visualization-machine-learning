
(*  \text{ReLU}(x) = \max(0, x) *)

relu[x_] := Piecewise[{{x, x >= 0}, {0, x < 0}}]

relu[x_] := Max[0, x]

Plot[relu[x], {x, -10, 10}, PlotRange -> All, AxesLabel -> {"x", "ReLU(x)"}]

(* 3D *)
relu2D[x_, y_] := Max[0, x + y]

Plot3D[relu2D[x, y], {x, -10, 10}, {y, -10, 10}, 
 PlotRange -> All, AxesLabel -> {"x", "y", "ReLU(x+y)"}, 
 ColorFunction -> "Rainbow", Mesh -> None]

