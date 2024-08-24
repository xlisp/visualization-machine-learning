## 1.普通最小二乘估计: 
('<-' (launch, (read.csv ("http://127.0.0.1:8003/challenger.csv", stringsAsFactors=FALSE))))
## Dataframe就是一张Excel表: 用Excel表大脑风暴特征
## 如一篇博客的特征表: 更新次数,创建时间,创建标题(需要降维),搜索标签,热度词个数(所有文章的词频前十), 热度词列表, 编辑时间列表(每次打开edit到提交update的时间) ==>>
## 线性回归,其它优化模型的组合 <==不断回归过去串联一切==> 优化特征工程
##    o_ring_ct distress_ct temperature pressure launch_id
## 1          6           0          66       50         1
## 2          6           1          70       50         2
## 3          6           0          69       50         3
## 4          6           0          68       50         4
## 5          6           0          67       50         5
## 6          6           0          72       50         6

                                        # _x 代表x的平均值
## temperature & distress_ct的协方差`Σ(x_i - _x)(y_i - _x)`
(cov (launch$temperature, launch$distress_ct)) #=>[1] -2.86166

## temperature的方差
(var (launch$temperature)) #=>  [1] 49.80237

## 平方误差最小的b值 a = _y - b_x
('<-' (b, ('/' ((cov (launch$temperature, launch$distress_ct)), (var (launch$temperature)))))) #=>[1] -0.05746032
## a的最优值
(('-' ((mean (launch$distress_ct)), (mean ('*' (b, ((launch$temperature))))))) -> a) #=>  [1] 4.301587

## 2. 相关系数
(('/' ((cov (launch$temperature, launch$distress_ct)), ('*' ((sd (launch$temperature)), (sd (launch$distress_ct)))) )) -> r) #=>  [1] -0.725671

## 内置的相关系数cor函数:
(cor (launch$temperature, launch$distress_ct)) # [1] -0.725671

## 3. 多元线性回归
((function (y, x, mx=(as.matrix (x)), cx=(cbind (Intercept=1, mx)))
    ('%*%' (('%*%' ((solve ('%*%' ((t (cx)), cx))), (t (cx)))), y)) ) -> reg)

(reg (y=(launch$distress_ct), x=(launch [3])))
##                    [,1]
## Intercept    4.30158730
## temperature -0.05746032

(reg (y=(launch$distress_ct), x=(launch [3:5])))
##                     [,1]
## Intercept    3.814247216
## temperature -0.055068768
## pressure     0.003428843
## launch_id   -0.016734090

## (library (magrittr))
## ((function (y, x, mx=(as.matrix (x)), cx=(cbind (Intercept=1, mx)))
##     ((solve ('%*%' ((t (cx)), cx)))
##         %>% (function (a) a)
##         ##%>% (function (a) ('%*%' ((t (cx)), a)))
##         ##%>% (function (a) ('%*%' (y, a)))
##     )) -> reg)
## 

(str (launch))
## 'data.frame':	23 obs. of  5 variables:
##  $ o_ring_ct  : int  6 6 6 6 6 6 6 6 6 6 ...
##  $ distress_ct: int  0 1 0 0 0 0 0 0 1 1 ...
##  $ temperature: int  66 70 69 68 67 72 73 70 57 63 ...
##  $ pressure   : int  50 50 50 50 50 50 100 100 200 200 ...
##  $ launch_id  : int  1 2 3 4 5 6 7 8 9 10 ...
## 

## 3. 例子: 应用线性回归预测医疗费用 ======================================================

((read.csv ("http://127.0.0.1:8003/insurance.csv", stringsAsFactors=FALSE)) -> insurance)
##      age    sex    bmi children smoker    region   charges
## 1     19 female 27.900        0    yes southwest 16884.924
## 2     18   male 33.770        1     no southeast  1725.552
## 3     28   male 33.000        3     no southeast  4449.462
## 4     33   male 22.705        0     no northwest 21984.471

(str (insurance))
## 'data.frame':	1338 obs. of  7 variables:
##  $ age     : int  19 18 28 33 32 31 46 37 37 60 ...
##  $ sex     : chr  "female" "male" "male" "male" ...
##  $ bmi     : num  27.9 33.8 33 22.7 28.9 ...
##  $ children: int  0 1 3 0 0 0 1 3 2 0 ...
##  $ smoker  : chr  "yes" "no" "no" "no" ...
##  $ region  : chr  "southwest" "southeast" "southeast" "northwest" ...
##  $ charges : num  16885 1726 4449 21984 3867 ...
## 

## 11111 ====>> charges是因变量或者是因子: 即医疗费用 => 预测医疗费用 ==>>> 看看它是如何分布的
(summary (insurance$charges)) #=>
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##    1122    4740    9382   13270   16640   63770 

## 画直方图:
## (hist (insurance$charges)) #==>> charges_hist.png

(table (insurance$region)) ##每个地区占的频率=>
## northeast northwest southeast southwest 
##       324       325       364       325 

## 3.1 探索特征之间的关系---相关系数矩阵
(cor (insurance [(c ("age", "bmi", "children", "charges"))]))
##                age       bmi   children    charges
## age      1.0000000 0.1092719 0.04246900 0.29900819
## bmi      0.1092719 1.0000000 0.01275890 0.19834097
## children 0.0424690 0.0127589 1.00000000 0.06799823
## charges  0.2990082 0.1983410 0.06799823 1.00000000

## age和age的关系是1,因为他们是同一个的
## charges和age和bmi(身体质量指数)的相关系数都是接近0.2的


## 3.2 可视化特征之间的关系------散点图矩阵
## (pairs (insurance [(c ("age", "bmi", "children", "charges"))])) #=> pairs_insurance.png
(library (psych)) ## pairs.panels可以显示拟合的线
## (pairs.panels (insurance [(c ("age", "bmi", "children", "charges"))])) #=> pairs_panels_insurance.png

## 3.3 基于数据训练模型 --------------
((lm (charges ~ age + children + bmi + sex + smoker + region, data=insurance)) -> ins_model)
## Call:
## lm(formula = charges ~ age + children + bmi + sex + smoker + 
##     region, data = insurance)
## 
## Coefficients:
##     (Intercept)              age         children              bmi  
##        -11938.5            256.9            475.5            339.2  
##         sexmale        smokeryes  regionnorthwest  regionsoutheast  
##          -131.3          23848.5           -353.0          -1035.0  
## regionsouthwest  
##          -960.1  
## 

## 3.4 评估模型的性能
(summary (ins_model))
## Call:
## lm(formula = charges ~ age + children + bmi + sex + smoker + 
##     region, data = insurance)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -11304.9  -2848.1   -982.1   1393.9  29992.8 
## 
## Coefficients:
##                 Estimate Std. Error t value Pr(>|t|)    
## (Intercept)     -11938.5      987.8 -12.086  < 2e-16 ***
## age                256.9       11.9  21.587  < 2e-16 ***
## children           475.5      137.8   3.451 0.000577 ***
## bmi                339.2       28.6  11.860  < 2e-16 ***
## sexmale           -131.3      332.9  -0.394 0.693348    
## smokeryes        23848.5      413.1  57.723  < 2e-16 ***
## regionnorthwest   -353.0      476.3  -0.741 0.458769    
## regionsoutheast  -1035.0      478.7  -2.162 0.030782 *  
## regionsouthwest   -960.0      477.9  -2.009 0.044765 *  
## ---
## Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
## 
## Residual standard error: 6062 on 1329 degrees of freedom
## Multiple R-squared:  0.7509,	Adjusted R-squared:  0.7494 
## F-statistic: 500.8 on 8 and 1329 DF,  p-value: < 2.2e-16
## 
