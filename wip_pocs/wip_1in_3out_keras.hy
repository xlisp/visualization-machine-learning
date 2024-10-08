;; keras function api + hylang: one input picture output predict age , name , man/woman.

;; HyLang is a Lisp dialect that runs on Python, so it can interoperate with Python libraries like TensorFlow/Keras. You can rewrite the Keras Functional API code into HyLang by preserving the same logic but translating the Python syntax into a more Lisp-like structure. Here's how the model definition and training process would look in HyLang:

;; ### Step 1: Import Libraries
;; In HyLang, importing modules is similar to Python but uses Lisp-style parentheses.

;; ```hy
(import [tensorflow.keras.layers [Input Conv2D MaxPooling2D Flatten Dense Dropout]]
        [tensorflow.keras.models [Model]]
        [tensorflow.keras.optimizers [Adam]]
        [tensorflow.keras.losses [MeanSquaredError SparseCategoricalCrossentropy BinaryCrossentropy]])
;; ```

;; ### Step 2: Define the Model
;; In HyLang, function definitions use the `defn` keyword, and the function body is written inside parentheses like Lisp.

;; ```hy
(defn create-model [input-shape num-names]
  ;; Input layer (Image)
  (setv inputs (Input :shape input-shape))

  ;; Shared Backbone (Convolutional layers)
  (setv x (-> inputs
              (Conv2D 64 :kernel-size [3 3] :activation 'relu' :padding 'same')
              (MaxPooling2D :pool-size [2 2])
              (Conv2D 128 :kernel-size [3 3] :activation 'relu' :padding 'same')
              (MaxPooling2D :pool-size [2 2])
              (Conv2D 256 :kernel-size [3 3] :activation 'relu' :padding 'same')
              (MaxPooling2D :pool-size [2 2])))

  ;; Flatten the feature maps
  (setv x (-> x
              (Flatten)
              (Dense 1024 :activation 'relu')))

  ;; Age Prediction Head (Regression)
  (setv age-output (Dense 1 :name 'age_output' x))  ;; Single output for age

  ;; Name Prediction Head (Classification)
  (setv name-output (Dense num-names :activation 'softmax' :name 'name_output' x))  ;; Multi-class output for name

  ;; Gender Prediction Head (Binary classification)
  (setv gender-output (Dense 1 :activation 'sigmoid' :name 'gender_output' x))  ;; Binary output (0 or 1)

  ;; Create model with inputs and three outputs
  (Model :inputs inputs :outputs [age-output name-output gender-output]))
;; ```

;; ### Step 3: Instantiate the Model
;; In HyLang, you can instantiate the model using the same logic as Python but using the Lisp syntax.

;; ```hy
(setv input-shape [64 64 3])
(setv num-names 100)  ;; Assuming 100 possible names for classification
(setv model (create-model input-shape num-names))
;; ```

;; ### Step 4: Compile the Model
;; Keras' `compile` function can also be called with keyword arguments in HyLang.

;; ```hy
(.compile model
          :optimizer (Adam :learning-rate 0.001)
          :loss {"age_output" (MeanSquaredError)
                 "name_output" (SparseCategoricalCrossentropy)
                 "gender_output" (BinaryCrossentropy)}
          :metrics {"age_output" ["mae"]  ;; Mean Absolute Error for age prediction
                    "name_output" ["accuracy"]  ;; Accuracy for name prediction
                    "gender_output" ["accuracy"]})  ;; Accuracy for gender prediction
;; ```

;; ### Step 5: Training the Model
;; HyLang uses `.` notation to call methods on objects, so fitting the model can be written like this:

;; ```hy
(setv history
      (.fit model
            train-images
            {"age_output" train-age-labels
             "name_output" train-name-labels
             "gender_output" train-gender-labels}
            :validation-data [val-images
                              {"age_output" val-age-labels
                               "name_output" val-name-labels
                               "gender_output" val-gender-labels}]
            :epochs 20
            :batch-size 32))
;; ```

;; ### Step 6: Inference (Predictions)
;; For inference, you can also call `.predict` on the model in HyLang:

;; ```hy
(setv predictions (.predict model test-image))

(setv predicted-age (first predictions 0 0))  ;; Age prediction
(setv predicted-name (first (.argmax (get predictions 1) 1)))  ;; Name prediction (index with max probability)
(setv predicted-gender (if (> (first predictions 2 0) 0.5) 1 0))  ;; Gender prediction (binary)

(print "Predicted Age: " predicted-age)
(print "Predicted Name ID: " predicted-name)
(print "Predicted Gender: " (if (= predicted-gender 0) "Man" "Woman"))
;; ```

;; ### Step 7: Visualizing Model Summary
;; You can also print the model summary in HyLang similarly to Python:

;; ```hy
(.summary model)
;; ```

;; ### Step 8: Customizing the Model (Optional)
;; If you want to add custom loss weights, you can do it the same way as in Python, just with HyLang syntax:

;; ```hy
(.compile model
          :optimizer (Adam :learning-rate 0.001)
          :loss {"age_output" (MeanSquaredError)
                 "name_output" (SparseCategoricalCrossentropy)
                 "gender_output" (BinaryCrossentropy)}
          :loss-weights {"age_output" 1.0
                         "name_output" 0.5
                         "gender_output" 0.5}
          :metrics ["accuracy"])
;; ```

;; In HyLang, everything is organized using S-expressions (Lisp-like parentheses), but the underlying TensorFlow/Keras logic remains the same. You can run this HyLang code if you have the necessary libraries installed and interoperate with Keras as in Python.
