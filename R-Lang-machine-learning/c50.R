## 1. 清洗数据 & factor
('<-' (credit, (read.csv ("http://127.0.0.1:8003/credit.csv"))))

(str (credit))
#=>
## 'data.frame':	1000 obs. of  21 variables:
##  $ checking_balance    : Factor w/ 4 levels "< 0 DM","> 200 DM",..: 1 3 4 1 1 4 4 3 4 3 ...
##  $ months_loan_duration: int  6 48 12 42 24 36 24 36 12 30 ...
##  ... ...

(table (credit$checking_balance))
##     < 0 DM   > 200 DM 1 - 200 DM    unknown 
##        274         63        269        394 

(table (credit$savings_balance))
##      < 100 DM     > 1000 DM  101 - 500 DM 501 - 1000 DM       unknown 
##           603            48           103            63           183 

(summary (credit$months_loan_duration))
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##     4.0    12.0    18.0    20.9    24.0    72.0 
## 

(summary (credit$amount))
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##     250    1366    2320    3271    3972   18424 

(table (credit$default))
##   1   2 
## 700 300 

## 在数据源的源头改变factor因子数据(!!!)
('<-' (credit$default, (factor (credit$default, levels=(c (1, 2)), labels=(c ("no", "yes"))))))


(set.seed (12345)) # 能够确保如果要重复这里的分析,那么可以获得相同的结果
(class ((runif (1000)))) # 1000个随机数, [1] "numeric", 产生0~1的随机数
## 2. 创建随机的训练数据集和测试数据集
('<-' (credit_rand, ((credit [(order ((runif (1000)))), ]))  ))
(str (credit_rand))
## 'data.frame':	1000 obs. of  21 variables:
##  $ checking_balance    : Factor w/ 4 levels "< 0 DM","> 200 DM",..: 4 3 3 3 3 1 1 1 1 4 ...
##  $ months_loan_duration: int  24 9 9 15 24 24 18 18 30 24 ...
##  $ default             : Factor w/ 2 levels "no","yes": NA NA NA NA NA NA NA NA NA NA ...
##  ... ...

(head (credit_rand$amount)) #=> [1] 2346 2030 1082 2631 3069 1333

## 大部分作为训练数据,小部分作为测试数据
('<-' (credit_train, (credit_rand [1:900, ])))
('<-' (credit_test, (credit_rand [901:1000, ])))

(round (('*' ((prop.table (table (credit_train$default))) ,100)), digits=1))
# ==>> factor因子数据替换成功
##  no  yes 
## 69.1 30.9 

(round (('*' ((prop.table (table (credit_test$default))) ,100)), digits=1))
## no yes 
## 78  22 

## 3. 基于c5.0算法来训练决策树=> 基于数据训练模型
(library (C50))
## [1] "C50"       "stats"     "graphics"  "grDevices" "utils"     "datasets" 
## [7] "methods"   "base"     

## ('<-' (credit_train$default, (factor (credit_train$default, levels=(c ("no", "yes")), labels=(c ("1", "2"))))))
# 如果没有这一步修改factor,调用C5.0函数就会报如下错误:
## Error in C5.0.default(credit_train[-17], credit_train$default) : 
##   C5.0 models require a factor outcome

('<-' (credit_model, (C5.0 ((credit_train [-17]), credit_train$default))))
## Call:
## C5.0.default(x = (credit_train[-17]), y = credit_train$default)
## 
## Classification Tree
## Number of samples: 900 
## Number of predictors: 20 
## 
## Tree size: 1 
## 
## Non-standard options: attempt to group attributes
## 

(summary (credit_model)) ###===>> 正确的结果是: c50_credit_model_summary.txt
#=> 因子数据没有替换的错误==>>
## Call:
## C5.0.default(x = (credit_train[-17]), y = credit_train$default)
## 
## 
## C5.0 [Release 2.07 GPL Edition]  	Sun Jul  2 10:52:54 2017
## -------------------------------
## 
## Class specified by attribute `outcome'
## *** ignoring cases with bad or unknown class
## 
## Read 0 cases (21 attributes) from undefined.data
## 
## Decision tree:
##  1 (0)
## 
## 
## Evaluation on training data (0 cases):
## 
## 	    Decision Tree   
## 	  ----------------  
## 	  Size      Errors  
## 
## 	     0    0( 0.0%)   <<
## 
## 
## 	   (a)   (b)    <-classified as
## 	  ----  ----
## 	                (a): class 1
## 	                (b): class 2
## 

## 4. 评估模型的性能
('<-' (credit_pred, (predict (credit_model, credit_test))))
## 因子数据没有替换导致的错误 ##=> address 0x10, cause 'memory not mapped'
## ==> ok的结果
##  [1] no  no  no  no  no  no  yes no  no  no  no  no  no  no  no  yes no  yes
## [19] no  no  no  no  no  no  no  no  no  no  no  no  yes yes yes no  yes no 
## [37] no  no  yes no  no  no  no  yes no  no  yes no  no  no  no  yes no  no 
## [55] no  no  no  yes no  no  no  no  no  no  no  no  no  yes no  no  no  no 
## [73] no  no  no  no  no  yes no  no  yes no  no  no  no  yes no  no  no  no 
## [91] no  no  no  yes no  no  yes no  no  no 
## Levels: no yes

(library (gmodels))

(CrossTable (credit_test$default, credit_pred, prop.chisq=FALSE, prop.c=FALSE, prop.r=FALSE, dnn=(c ('actual default', 'predicted default'))))

 
##     Cell Contents
##  |-------------------------|
##  |                       N |
##  |         N / Table Total |
##  |-------------------------|
##  
##   
##  Total Observations in Table:  100 
##  
##   
##                 | predicted default 
##  actual default |        no |       yes | Row Total | 
##  ---------------|-----------|-----------|-----------|
##              no |        67 |        11 |        78 | 
##                 |     0.670 |     0.110 |           | 
##  ---------------|-----------|-----------|-----------|
##             yes |        15 |         7 |        22 | 
##                 |     0.150 |     0.070 |           | 
##  ---------------|-----------|-----------|-----------|
##    Column Total |        82 |        18 |       100 | 
##  ---------------|-----------|-----------|-----------|
##  
##   
##  $t
##       y
##  x     no yes
##    no  67  11
##    yes 15   7
##  
##  $prop.row
##       y
##  x            no       yes
##    no  0.8589744 0.1410256
##    yes 0.6818182 0.3181818
##  
##  $prop.col
##       y
##  x            no       yes
##    no  0.8170732 0.6111111
##    yes 0.1829268 0.3888889
##  
##  $prop.tbl
##       y
##  x       no  yes
##    no  0.67 0.11
##    yes 0.15 0.07
##  
