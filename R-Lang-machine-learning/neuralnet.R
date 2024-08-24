## 用人工神经网络对混凝土的强度进行建模 ==>> 预测混凝土的强度
## 1. 清洗数据格式化
('<-' (concrete, (read.csv ("http://127.0.0.1:8003/concrete.csv", stringsAsFactors=FALSE))))

(str (concrete))
## ===>>
## 'data.frame':	1030 obs. of  9 variables:
##  $ cement      : num  540 540 332 332 199 ...
##  $ slag        : num  0 0 142 142 132 ...
##  $ ash         : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ water       : num  162 162 228 228 192 228 228 228 228 228 ...
##  $ superplastic: num  2.5 2.5 0 0 0 0 0 0 0 0 ...
##  $ coarseagg   : num  1040 1055 932 932 978 ...
##  $ fineagg     : num  676 676 594 594 826 ...
##  $ age         : int  28 28 270 365 360 90 365 28 28 28 ...
##  $ strength    : num  80 61.9 40.3 41 44.3 ...
## 

('<-' (normalize,
  (function (x)
    ('/' (('-' (x, (min (x)))),
    ('-' ((max (x)), (min (x)))))))))

('<-' (concrete_norm, (as.data.frame ((lapply (concrete, normalize))))))

(summary (concrete_norm$strength))
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##  0.0000  0.2664  0.4001  0.4172  0.5457  1.0000 

## normalize标准化数据之后
(summary (concrete$strength))
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##    2.33   23.71   34.45   35.82   46.13   82.60 
## 

('<-' (concrete_train, (concrete_norm [1:773, ])))
('<-' (concrete_test, (concrete_norm [774:1030, ])))

## 2. 基于数据训练模型
(library (neuralnet))
## [1] "neuralnet" "stats"     "graphics"  "grDevices" "utils"     "datasets" 
## [7] "methods"   "base"     

## neuralnet函数用于数值预测的神经网络: 多种原料=>强度预测, 用多层前馈神经网络
##(neuralnet ('+' ((strength ~ cement), slag))
('<-' (concrete_model, (neuralnet (strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age, data=concrete_train))))
## ===>>>
## $response
##          strength
## 1   0.96748473901
## 2   0.74199576430
## ... ...
## $result.matrix
## 1
## error                       5.667653807531
## reached.threshold           0.009965416362
## steps                    2264.000000000000
## Intercept.to.1layhid1      -0.576176258074
## cement.to.1layhid1          3.083416101750
## slag.to.1layhid1            1.333140949639
## ash.to.1layhid1             0.293304656065
## water.to.1layhid1          -2.490013193154
## superplastic.to.1layhid1    2.425785903929
## coarseagg.to.1layhid1      -0.338636250824
## fineagg.to.1layhid1        -1.664347763113
## age.to.1layhid1            12.448727298701
## Intercept.to.strength       0.044186970773
## 1layhid.1.to.strength       0.630335882646
## 
## attr(,"class")
## [1] "nn"
## 
## (plot (concrete_model)) #===>> concrete_model_plot.png 网络拓扑结构图

## 2. 评估模型的性能
## 用compute生成预测, 返回一个带有两个分量的列表: $neurons用来存储网络中每一层的神经元, $net.results用来存储预测值
('<-' (model_results, (compute (concrete_model, (concrete_test [1:8])))))
## $neurons
## $neurons[[1]]
##      1        cement          slag          ash         water  superplastic
## 774  1 0.63926940639 0.00000000000 0.0000000000 0.51277955272 0.00000000000
## ...
## 1030 0.4019830262
## 
## 预测predicted强度
('<-' (predicted_strength, (model_results$net.result)))
##              [,1]
## 774  0.3898173578
## 775  0.2429694647
## 776  0.2506441174

## doc: cor用来获取两个数值向量之间的相关性
(cor (predicted_strength, concrete_test$strength))
##              [,1]
## [1,] 0.7195218932
## 
