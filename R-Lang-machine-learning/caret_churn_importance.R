## 1. churnTrain数据准备
(library (C50))
(data (churn))

((churnTrain [,! (names (churnTrain)) %in% (c ("state", "area_code", "account_length"))]) -> churnTrain)
##      international_plan voice_mail_plan number_vmail_messages total_day_minutes
## 1                    no             yes                    25             265.1
## 2                    no             yes                    26             161.6
## 3                    no              no                     0             243.4

(set.seed (2))
((sample (2, (nrow (churnTrain)), replace=TRUE, prob=(c (0.7, 0.3)))) -> ind)
##    [1] 1 2 1 1 2 2 1 2 1 1 1 1 2 1 1 2 2 1 1 1 1 1 2 1 1 1 1 1 2 1 1 1 2 2 1 1 2
##   [38] 1 1 1 2 1 1 1 2 2 2 1 1 2 1 1 1 2 1 2 2 2 1 2 2 2 1 1 2 1 1 1 1 1 1 1 1 1
## 

((churnTrain [('==' (ind, 1)),]) -> trainset)
((churnTrain [('==' (ind, 2)),]) -> testset)

(dim (trainset)) #=>  [1] 2308   17
(dim (testset)) #=>  [1] 1025   17

## 2. caret特征挖掘高明之处
(library (caret))
(library (rpart))
(library (e1071)) ## 一切都是向量矩阵,很有Lisp的基因
##  [1] "e1071"     "rpart"     "caret"     "ggplot2"   "lattice"   "C50"      
##  [7] "stats"     "graphics"  "grDevices" "utils"     "datasets"  "methods"  
## [13] "base"     

## repeats=3, for 3次
((trainControl (method="repeatedcv", number=10,repeats=3)) -> control)
((train (churn~., data=trainset, method="rpart",preProcess="scale", trControl=control)) -> model)
## CART 
## 2315 samples
##   16 predictor
##    2 classes: 'yes', 'no' 
## Pre-processing: scaled (16) 
## Resampling: Cross-Validated (10 fold, repeated 3 times) 
## Summary of sample sizes: 2084, 2084, 2083, 2083, 2082, 2084, ... 
## Resampling results across tuning parameters:
##   cp          Accuracy   Kappa    
##   0.05555556  0.8995112  0.5174059
##   0.07456140  0.8593389  0.2124126
##   0.07602339  0.8567440  0.1898221
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.05555556.
## 

## 3. 画重要等级的图
((varImp (model, scale=FALSE)) -> importance)
## rpart variable importance
##                               Overall
## number_customer_service_calls 116.015
## total_day_minutes             106.988
## total_day_charge              100.648
## international_planyes          86.789
## voice_mail_planyes             25.974
## total_eve_charge               23.097
## total_eve_minutes              23.097
## number_vmail_messages          19.885
## total_intl_minutes              6.347
## total_day_calls                 0.000
## total_night_charge              0.000
## total_night_minutes             0.000
## total_eve_calls                 0.000
## total_night_calls               0.000
## total_intl_calls                0.000
## total_intl_charge               0.000
## 

## (plot (importance)) ##=> fs_churn_importance_by_caret.png





## =======
(str (churnTrain))
## 'data.frame':	3333 obs. of  20 variables:
##  $ state                        : Factor w/ 51 levels "AK","AL","AR",..: 17 36 32 36 37 2 20 25 19 50 ...
##  $ account_length               : int  128 107 137 84 75 118 121 147 117 141 ...
##  $ area_code                    : Factor w/ 3 levels "area_code_408",..: 2 2 2 1 2 3 3 2 1 2 ...
##  $ international_plan           : Factor w/ 2 levels "no","yes": 1 1 1 2 2 2 1 2 1 2 ...
##  $ voice_mail_plan              : Factor w/ 2 levels "no","yes": 2 2 1 1 1 1 2 1 1 2 ...
##  $ number_vmail_messages        : int  25 26 0 0 0 0 24 0 0 37 ...
##  $ total_day_minutes            : num  265 162 243 299 167 ...
##  $ total_day_calls              : int  110 123 114 71 113 98 88 79 97 84 ...
##  $ total_day_charge             : num  45.1 27.5 41.4 50.9 28.3 ...
##  $ total_eve_minutes            : num  197.4 195.5 121.2 61.9 148.3 ...
##  $ total_eve_calls              : int  99 103 110 88 122 101 108 94 80 111 ...
##  $ total_eve_charge             : num  16.78 16.62 10.3 5.26 12.61 ...
##  $ total_night_minutes          : num  245 254 163 197 187 ...
##  $ total_night_calls            : int  91 103 104 89 121 118 118 96 90 97 ...
##  $ total_night_charge           : num  11.01 11.45 7.32 8.86 8.41 ...
##  $ total_intl_minutes           : num  10 13.7 12.2 6.6 10.1 6.3 7.5 7.1 8.7 11.2 ...
##  $ total_intl_calls             : int  3 3 5 7 3 6 7 6 4 5 ...
##  $ total_intl_charge            : num  2.7 3.7 3.29 1.78 2.73 1.7 2.03 1.92 2.35 3.02 ...
##  $ number_customer_service_calls: int  1 1 0 2 3 0 3 0 1 0 ...
##  $ churn                        : Factor w/ 2 levels "yes","no": 2 2 2 2 2 2 2 2 2 2 ...
## 
