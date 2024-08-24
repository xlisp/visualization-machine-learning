
('<-' (wbcd, (read.csv ("http://127.0.0.1:8003/wisc_bc_data.csv", stringsAsFactors=FALSE))))
#            id diagnosis radius_mean texture_mean perimeter_mean area_mean
# 1      842302         M      17.990        10.38         122.80    1001.0
# 2      842517         M      20.570        17.77         132.90    1326.0

('<-' (wbcd, (wbcd [-1]))) # 去除第一列

(table (wbcd$diagnosis))
#   B   M  # B是良性肿块, B是恶性肿块
# 357 212 

## 1. 两组特征训练数据: B & M =>> 预测新的数据是B还是M
### 替换一下数据名称 ==>> 因子的意义
('<-' (wbcd$diagnosis, (factor (wbcd$diagnosis, levels=(c ("B", "M")), labels=(c ("良性肿块", "恶性肿块"))))))
##     diagnosis radius_mean texture_mean perimeter_mean area_mean smoothness_mean
## 1    恶性肿块      17.990        10.38         122.80    1001.0         0.11840
## 2    恶性肿块      20.570        17.77         132.90    1326.0         0.08474
## 21   良性肿块      13.080        15.71          85.63     520.0         0.10750
##

(round (('*' ((prop.table (table (wbcd$diagnosis))) ,100)), digits=1))
## 2.良性肿块 恶性肿块 
##     62.7     37.3   # 百分比计算

## 3.总结特征, 细胞核的3种特征: 最小, 最大, 平均值,中间值等
(summary ((wbcd [(c ("radius_mean", "area_mean", "smoothness_mean"))])))
##   radius_mean       area_mean      smoothness_mean  
##  Min.   : 6.981   Min.   : 143.5   Min.   :0.05263  
##  1st Qu.:11.700   1st Qu.: 420.3   1st Qu.:0.08637  
##  Median :13.370   Median : 551.1   Median :0.09587  
##  Mean   :14.127   Mean   : 654.9   Mean   :0.09636  
##  3rd Qu.:15.780   3rd Qu.: 782.7   3rd Qu.:0.10530  
##  Max.   :28.110   Max.   :2501.0   Max.   :0.16340  

## 4. 标准化数值型数据,以便确保在标准的范围内 (normalize ((c (1, 2, 3, 4, 5)))) #=>  [1] 0.00 0.25 0.50 0.75 1.00
### (normalize ((c (10, 20, 30, 40, 50)))) #=>  [1] 0.00 0.25 0.50 0.75 1.00
('<-' (normalize,
  (function (x)
    ('/' (('-' (x, (min (x)))),
      ('-' ((max (x)), (min (x)))))))))

## 5.运用normalize标准化所有的表格数据: lapply相当于each
(wbcd [2:31])
##     radius_mean texture_mean perimeter_mean area_mean smoothness_mean
## 1        17.990        10.38         122.80    1001.0         0.11840
## 2        20.570        17.77         132.90    1326.0         0.08474
(lapply ((wbcd [2:31]), normalize)) # 对于每一列的所有值都执行normalize,相当于map了

('<-' (wbcd_n, (as.data.frame ((lapply ((wbcd [2:31]), normalize)))))) # 重新变成data.frame
##     radius_mean texture_mean perimeter_mean  area_mean smoothness_mean
## 1    0.52103744   0.02265810     0.54598853 0.36373277      0.59375282
## 2    0.64314449   0.27257355     0.61578329 0.50159067      0.28987993
## 

(summary (wbcd_n$area_mean))
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##  0.0000  0.1174  0.1729  0.2169  0.2711  1.0000 
##

## 6. 创建训练数据集 和 测试数据集: 用同一个数据源拆分出来
('<-' (wbcd_train, (wbcd_n [1:469, ]))) ## 大部分的数据作为训练数据
('<-' (wbcd_test, (wbcd_n [470:569, ]))) ## 小部分作为测试数据

('<-' (wbcd_train_labels, (wbcd [1:469, 1]))) ### [129] 良性肿块 恶性肿块 良性肿块 ...
('<-' (wbcd_test_labels, (wbcd [470:569, 1]))) ##  [49] 恶性肿块 良性肿块 良性肿块 ...

## 7. kNN数据训练模型 111111
(library (class))
## [1] "class"     "stats"     "graphics"  "grDevices" "utils"     "datasets" 
## [7] "methods"   "base"

('<-' (wbcd_test_pred, (knn (train=wbcd_train, test=wbcd_test, cl=wbcd_train_labels, k=21))))
#### --->>> knn返回wbcd_test_pred因子向量,为测试数据集中的每一个案例返回一个预测标签
##   [1] 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块
##   [9] 良性肿块 良性肿块 恶性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块
##  [17] 良性肿块 良性肿块 恶性肿块 良性肿块 良性肿块 良性肿块 良性肿块 恶性肿块
##  [25] 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 恶性肿块 恶性肿块 良性肿块
##  [33] 恶性肿块 良性肿块 恶性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块
##  [41] 恶性肿块 良性肿块 良性肿块 恶性肿块 良性肿块 良性肿块 良性肿块 恶性肿块
##  [49] 恶性肿块 良性肿块 良性肿块 良性肿块 恶性肿块 良性肿块 良性肿块 良性肿块
##  [57] 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块
##  [65] 恶性肿块 良性肿块 恶性肿块 恶性肿块 良性肿块 良性肿块 良性肿块 良性肿块
##  [73] 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块
##  [81] 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块
##  [89] 良性肿块 良性肿块 良性肿块 良性肿块 良性肿块 恶性肿块 恶性肿块 恶性肿块
##  [97] 恶性肿块 恶性肿块 恶性肿块 良性肿块
## Levels: 良性肿块 恶性肿块
## 

## 8. 评估模型的性能 ==>> perl&C++的chaocore
(library (gmodels))
(CrossTable (x=wbcd_test_labels, y=wbcd_test_pred, prop.chisq=FALSE))

##       
##         Cell Contents
##      |-------------------------|
##      |                       N |
##      |           N / Row Total |
##      |           N / Col Total |
##      |         N / Table Total |
##      |-------------------------|
##      
##       
##      Total Observations in Table:  100 
##      
##       
##                       | wbcd_test_pred 
##      wbcd_test_labels |  良性肿块 |  恶性肿块 | Row Total | 
##      -----------------|-----------|-----------|-----------|
##              良性肿块 |        77 |         0 |        77 | 
##                       |     1.000 |     0.000 |     0.770 | 
##                       |     0.975 |     0.000 |           | 
##                       |     0.770 |     0.000 |           | 
##      -----------------|-----------|-----------|-----------|
##              恶性肿块 |         2 |        21 |        23 | 
##                       |     0.087 |     0.913 |     0.230 | 
##                       |     0.025 |     1.000 |           | 
##                       |     0.020 |     0.210 |           | 
##      -----------------|-----------|-----------|-----------|
##          Column Total |        79 |        21 |       100 | 
##                       |     0.790 |     0.210 |           | 
##      -----------------|-----------|-----------|-----------|
##      
##       
##      $t
##                y
##      x          良性肿块 恶性肿块
##        良性肿块       77        0
##        恶性肿块        2       21
##      
##      $prop.row
##                y
##      x            良性肿块   恶性肿块
##        良性肿块 1.00000000 0.00000000
##        恶性肿块 0.08695652 0.91304348
##      
##      $prop.col
##                y
##      x            良性肿块   恶性肿块
##        良性肿块 0.97468354 0.00000000
##        恶性肿块 0.02531646 1.00000000
##      
##      $prop.tbl
##                y
##      x          良性肿块 恶性肿块
##        良性肿块     0.77     0.00
##        恶性肿块     0.02     0.21
##      
