
FileExistsQ["/Users/emacspy/EmacsPyPro/emacspy-machine-learning/mnist_model.onnx"]

model = ONNXModelImport[
  "/Users/emacspy/EmacsPyPro/emacspy-machine-learning/mnist_model.onnx"]

preprocessImage[image_] := Module[
  {resizedImage, grayscaleImage, normalizedImage},
  
  (* Resize the image to 28x28 pixels *)
  resizedImage = ImageResize[image, {28, 28}];
  
  (* Convert the image to grayscale *)
  grayscaleImage = ColorConvert[resizedImage, "Grayscale"];
  
  (* Normalize the image with the same mean and std as MNIST *)
  normalizedImage = (ImageData[grayscaleImage] - 0.1307)/0.3081;
  
  (* Return the preprocessed image as a tensor *)
  normalizedImage
]

recognizeDigit[image_] := Module[
  {processedImage, prediction},
  
  (* Preprocess the image *)
  processedImage = preprocessImage[image];
  
  (* Perform inference using the model *)
  prediction = model[processedImage];
  
  (* Return the digit with the highest probability *)
  First[Ordering[prediction, -1]]
]

(* Example usage *)
imagePath = "path_to_your_handwritten_digit_image.png";
inputImage = Import[imagePath];
predictedDigit = recognizeDigit[inputImage];
Print["Predicted Digit: ", predictedDigit];

