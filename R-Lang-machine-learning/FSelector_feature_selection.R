## ======== FSelector 特征筛选 ===============
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

## 2.
(library (FSelector))
## [1] "FSelector" "stats"     "graphics"  "grDevices" "utils"     "datasets" 
## [7] "methods"   "base"     

## 计算每个属性的权值
((random.forest.importance (churn~., trainset, importance.type=1)) -> weights)
##                               attr_importance
## international_plan                 95.3025240
## voice_mail_plan                    26.9096412
## number_vmail_messages              31.3393068
## total_day_minutes                  52.1119553
## total_day_calls                     0.7384755
## total_day_charge                   55.7678957
## total_eve_minutes                  32.1862070
## total_eve_calls                    -1.3696832
## total_eve_charge                   32.4271470
## total_night_minutes                23.2257984
## total_night_calls                   3.6447715
## total_night_charge                 23.0453713
## total_intl_minutes                 34.0599517
## total_intl_calls                   50.3540726
## total_intl_charge                  33.1153116
## number_customer_service_calls     111.3517996
## 

## 获取权重最高的5个属性
((cutoff.k (weights, 5)) -> subset)
## [1] "number_customer_service_calls" "international_plan"           
## [3] "total_day_charge"              "total_day_minutes"            
## [5] "total_intl_calls"             
## 

(as.simple.formula (subset, "Class"))
## Class ~ number_customer_service_calls + international_plan + 
##     total_day_charge + total_day_minutes + total_intl_calls
## <environment: 0x7f94bab619f0>
## 
