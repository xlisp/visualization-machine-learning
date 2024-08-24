
("http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/" -> root)

((read.table (paste (root, "madelon_train.data", sep = ""))) -> predictors)

((read.table (paste (root, "madelon_train.labels", sep = ""))) -> decision)

((data.frame (predictors, decision=(factor (decision [, 1])))) -> Madelon)

(set.seed (7777))

(library (Boruta))

((Boruta (decision ~ ., data=Madelon)) -> Boruta.Madelon) ## 算了十多分钟
## Computing permutation importance.. Progress: 96%. Estimated remaining time: 1 seconds.
## Computing permutation importance.. Progress: 86%. Estimated remaining time: 5 seconds.
## Computing permutation importance.. Progress: 99%. Estimated remaining time: 0 seconds.
## Computing permutation importance.. Progress: 88%. Estimated remaining time: 4 seconds.
## 
## Boruta performed 99 iterations in 11.12934 mins.
##  20 attributes confirmed important: V106, V129, V154, V242, V282 and 15
## more;
##  479 attributes confirmed unimportant: V1, V10, V100, V101, V102 and
## 474 more;
##  1 tentative attributes left: V335;
## 

##(plotZHistory (Boruta.Madelon)) #=> 没有"plotZHistory"这个函数
##(plotImpHistory (Boruta.Madelon)) #=> 
