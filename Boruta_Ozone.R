(library (Boruta))
(library (mlbench))
## [1] "mlbench"   "stats"     "graphics"  "grDevices" "utils"     "datasets"
## [7] "methods"   "base"

(data ("Ozone")) #=>  [1] "Ozone", dataframe
#     V1 V2 V3 V4   V5 V6 V7 V8    V9  V10 V11   V12 V13
# 5    1  5  1  5 5760  3 51 54 45.32 1450  25 57.02  60
# 6    1  6  2  6 5720  4 69 35 49.64 1568  15 53.78  60
# 7    1  7  3  4 5790  6 19 45 46.40 2631 -33 54.14 100
# 8    1  8  4  4 5790  3 25 55 52.70  554 -28 64.76 250
# 9    1  9  5  6 5700  3 73 41 48.02 2083  23 52.52 120

((na.omit (Ozone)) -> Ozone)

(set.seed (1))

((Boruta (V4 ~ ., data=Ozone, doTrace=2, ntree=500)) -> Boruta.Ozone)
##  1. run of importance source...
##  11. run of importance source...
## After 11 iterations, +1.3 secs: 
##  confirmed 9 attributes: V1, V10, V11, V12, V13 and 4 more;
##  rejected 2 attributes: V2, V3;
##  still have 1 attribute left.
## 
##  12. run of importance source...
##  21. run of importance source...
## After 21 iterations, +2.2 secs: 
##  rejected 1 attribute: V6;
##  no more attributes left.
## 
## Boruta performed 21 iterations in 2.155657 secs.
##  9 attributes confirmed important: V1, V10, V11, V12, V13 and 4 more;
##  3 attributes confirmed unimportant: V2, V3, V6;
## 

## (plot (Boruta.Ozone)) =>> Boruta_Ozone.png


(set.seed (1))
((Boruta (V4 ~ ., data=Ozone, maxRuns=12)) -> Boruta.Short)
## Boruta performed 11 iterations in 0.9771039 secs.
##  9 attributes confirmed important: V1, V10, V11, V12, V13 and 4 more;
##  2 attributes confirmed unimportant: V2, V3;
##  1 tentative attributes left: V6;
## 

(TentativeRoughFix (Boruta.Short)) ## 同上的输出结果

(getConfirmedFormula (Boruta.Ozone))
## V4 ~ V1 + V5 + V7 + V8 + V9 + V10 + V11 + V12 + V13
## <environment: 0x7f8831a29478>

(attStats (Boruta.Ozone))
##        meanImp  medianImp     minImp     maxImp   normHits  decision
## V1   9.3406946  9.1881521  8.0645606 11.0689104 1.00000000 Confirmed
## V2   1.0693477  0.9569150 -0.7559703  2.9620623 0.16666667  Rejected
## V3  -0.9873096 -0.9355856 -2.6827062  0.5231361 0.00000000  Rejected
## V5   9.2088237  9.2421518  7.6002832 10.3374204 1.00000000 Confirmed
## V6   1.1733677  1.0083335 -0.3135760  3.3103123 0.04166667  Rejected
## V7  11.4760933 11.9068407  9.7520110 13.6941090 1.00000000 Confirmed
## V8  17.0992159 17.3075209 15.3788576 18.5088471 1.00000000 Confirmed
## V9  19.2203739 19.1062824 17.8229409 21.0290770 1.00000000 Confirmed
## V10 10.0826844 10.3426920  8.3303450 11.5526271 1.00000000 Confirmed
## V11 12.2283924 12.2067689 10.8172120 13.8378429 1.00000000 Confirmed
## V12 14.6852344 14.6627875 13.2286016 15.9460136 1.00000000 Confirmed
## V13  9.4474500  9.4095265  7.5137362 10.7496453 1.00000000 Confirmed
## 

(str (Ozone))
##  'data.frame':   203 obs. of  13 variables:
##   $ V1 : Factor w/ 12 levels "1","2","3","4",..: 1 1 1 1 1 1 1 1 1 1 ...
##   $ V2 : Factor w/ 31 levels "1","2","3","4",..: 5 6 7 8 9 12 13 14 15 16 ...
##   $ V3 : Factor w/ 7 levels "1","2","3","4",..: 1 2 3 4 5 1 2 3 4 5 ...
##   $ V4 : num  5 6 4 4 6 6 5 4 4 7 ...
##   $ V5 : num  5760 5720 5790 5790 5700 5720 5760 5780 5830 5870 ...
##   $ V6 : num  3 4 6 3 3 3 6 6 3 2 ...
##   $ V7 : num  51 69 19 25 73 44 33 19 19 19 ...
##   $ V8 : num  54 35 45 55 41 51 51 54 58 61 ...
##   $ V9 : num  45.3 49.6 46.4 52.7 48 ...
##   $ V10: num  1450 1568 2631 554 2083 ...
##   $ V11: num  25 15 -33 -28 23 9 -44 -44 -53 -67 ...
##   $ V12: num  57 53.8 54.1 64.8 52.5 ...
##   $ V13: num  60 60 100 250 120 150 40 200 250 200 ...
##   - attr(*, "na.action")=Class 'omit'  Named int [1:163] 1 2 3 4 10 11 17 18 20 24 ...
##    .. ..- attr(*, "names")= chr [1:163] "1" "2" "3" "4" ...
## 

## ------>>> 生成特征图谱的数据分析 -------->>> Boruta_Ozone.png
## 最右边的数据: 特征最重要的数据
(summary (Ozone$V9))
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##   27.68   49.64   56.48   56.54   66.20   82.58

## 最左边的数据: 特征最不明显的数据
(summary (Ozone$V2))
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
##  5  7  7  4  6  7  7  7  9  6  5  9  8  8  8  7  7  6  6  6  7  8  8  5  4  6
## 27 28 29 30 31
##  8  6  7  7  2
##

(Ozone$V2) # 太多的Levels了
## [201] 28 29 30
## 31 Levels: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ... 31
##

# 倒数第一的绿色:
(summary (Ozone$V1))
##  1  2  3  4  5  6  7  8  9 10 11 12
## 17 17 21 21 10 17 13 15 16 18 17 21
##

(Ozone$V1)
## [201] 12 12 12
## Levels: 1 2 3 4 5 6 7 8 9 10 11 12
##

