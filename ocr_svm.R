
('<-' (letters, (read.csv ("http://127.0.0.1:8003/letterdata.csv"))))

(str (letters)) ##=>
## 'data.frame':	20000 obs. of  17 variables:
##  $ letter: Factor w/ 26 levels "A","B","C","D",..: 20 9 4 14 7 19 2 1 10 13 ...
##  $ xbox  : int  2 5 4 7 2 4 4 1 2 11 ...
##  $ ybox  : int  8 12 11 11 1 11 2 1 2 15 ...
##  $ width : int  3 3 6 6 3 5 5 3 4 13 ...
##  $ height: int  5 7 8 6 1 8 4 2 4 9 ...
##  $ onpix : int  1 2 6 3 1 3 4 1 2 7 ...
##  $ xbar  : int  8 10 10 5 8 8 8 8 10 13 ...
##  $ ybar  : int  13 5 6 9 6 8 7 2 6 2 ...
##  $ x2bar : int  0 5 2 4 6 6 6 2 2 6 ...
##  $ y2bar : int  6 4 6 6 6 9 6 2 6 2 ...
##  $ xybar : int  6 13 10 4 6 5 7 8 12 12 ...
##  $ x2ybar: int  10 3 3 4 5 6 6 2 4 1 ...
##  $ xy2bar: int  8 9 7 10 9 6 6 8 8 9 ...
##  $ xedge : int  0 2 3 6 1 0 2 1 1 8 ...
##  $ xedgey: int  8 8 7 10 7 8 8 6 6 1 ...
##  $ yedge : int  0 4 3 2 5 9 7 2 1 1 ...
##  $ yedgex: int  8 10 9 8 10 7 10 7 7 8 ...

('<-' (letters_train, (letters [1:16000, ])))
('<-' (letters_test, (letters [16001:20000, ])))

## 2. 数据训练模型
(library (kernlab))
## [1] "kernlab"   "stats"     "graphics"  "grDevices" "utils"     "datasets" 
## [7] "methods"   "base"     

('<-' (letter_classifier, (ksvm (letter ~ ., data=letters_train, kernel="vanilladot"))))
##  Setting default kernel parameters  
## Support Vector Machine object of class "ksvm" 
## 
## SV type: C-svc  (classification) 
##  parameter : cost C = 1 
## 
## Linear (vanilla) kernel function. 
## 
## Number of Support Vectors : 7037 
## 
## Objective Function Value : -14.1746 -20.0072 -23.5628 -6.2009 -7.5524 -32.7694 .....
## Training error : 0.130062
## 

## 3. 评估模型的性能

('<-' (letter_predictions, (predict (letter_classifier, letters_test))))
##   [1] U N V X N H E Y G E N B G L G W M D Y R P D E W D S R G V R D J C I T C N
## ... ...
## [3997] C T S A
## Levels: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z

(head (letter_predictions)) #=> 前6个预测的字母是 U N V X N H
## [1] U N V X N H
## Levels: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
## 
## 预测的值和真实的值进行比较=>
(table (letter_predictions, letters_test$letter))
## letter_predictions   A   B   C   D   E   F   G   H   I   J   K   L   M   N   O
##                  A 144   0   0   0   0   0   0   0   0   1   0   0   1   2   2
##                  B   0 121   0   5   2   0   1   2   0   0   1   0   1   0   0
##                  C   0   0 120   0   4   0  10   2   2   0   1   3   0   0   2
##                  D   2   2   0 156   0   1   3  10   4   3   4   3   0   5   5
## 
## ... ... ... 
##                  X   0   0   0   1   0   0   0   0 137   1   1
##                  Y   7   0   0   0   3   0   0   0   0 127   0
##                  Z   0   0   0  18   3   0   0   0   0   0 132
## 

(round (('*' ((prop.table (table ('==' (letter_predictions, letters_test$letter)))) ,100)), digits=1))
## ==>> 正确率为83.9%
## FALSE  TRUE 
##  16.1  83.9 
## 
