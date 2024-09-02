
featureMatrix = {
  {1.0, 0.5, 0.2},
  {0.3, 0.8, 0.6},
  {0.7, 0.1, 0.4}
};

MatrixPlot[featureMatrix]

MatrixPlot[featureMatrix, ColorFunction -> "Rainbow", Frame -> False, Mesh -> True]

(* bug *)
MatrixPlot[
  featureMatrix,
  FrameTicks -> {{"Row 1", "Row 2", "Row 3"}, {"Feature 1", "Feature 2", "Feature 3"}},
  ColorFunction -> "Rainbow"
]
