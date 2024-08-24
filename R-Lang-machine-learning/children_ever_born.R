((read.table ("http://data.princeton.edu/wws509/datasets/ceb.dat")) -> ceb)

(str (ceb))
## 'data.frame':	70 obs. of  7 variables:
##  $ dur : Factor w/ 6 levels "0-4","10-14",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ res : Factor w/ 3 levels "rural","Suva",..: 2 2 2 2 3 3 3 3 1 1 ...
##  $ educ: Factor w/ 4 levels "lower","none",..: 2 1 4 3 2 1 4 3 2 1 ...
##  $ mean: num  0.5 1.14 0.9 0.73 1.17 0.85 1.05 0.69 0.97 0.96 ...
##  $ var : num  1.14 0.73 0.67 0.48 1.06 1.59 0.73 0.54 0.88 0.81 ...
##  $ n   : int  8 21 42 51 12 27 39 51 62 102 ...
##  $ y   : num  4 23.9 37.8 37.2 14 ...

## 泊松分布用直方图表示: 给响应变量 育子数 做直方图，可以清楚看到其偏倚度
(hist (ceb$y, breaks = 50, xlab = "children ever born", main = "Distribution of CEB"))
## => number_of_children_ever_born_poisson.png

## The cell number (1 to 71, cell 68 has no observations), 
## “dur” = marriage duration (1=0-4, 2=5-9, 3=10-14, 4=15-19, 5=20-24, 6=25-29), 
## “res” = residence (1=Suva, 2=Urban, 3=Rural), 
## “educ” = education (1=none, 2=lower primary, 3=upper primary, 4=secondary+), 
## “mean” = mean number of children ever born (e.g. 0.50), 
## “var” = variance of children ever born (e.g. 1.14), and 
## “n” = number of women in the cell (e.g. 8), 
## “y” = number of children ever born.

((glm (y ~ educ + res + dur, offset = log(n), family = poisson(), data = ceb)) -> fit)

(summary (fit))
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.2912  -0.6649   0.0759   0.6606   3.6790  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)  0.05695    0.04805   1.185    0.236    
## educnone    -0.02308    0.02266  -1.019    0.308    
## educsec+    -0.33266    0.05388  -6.174 6.67e-10 ***
## educupper   -0.12475    0.03000  -4.158 3.21e-05 ***
## resSuva     -0.15122    0.02833  -5.338 9.37e-08 ***
## resurban    -0.03896    0.02462  -1.582    0.114    
## dur10-14     1.37053    0.05108  26.833  < 2e-16 ***
## dur15-19     1.61423    0.05121  31.524  < 2e-16 ***
## dur20-24     1.78549    0.05122  34.856  < 2e-16 ***
## dur25-29     1.97679    0.05005  39.500  < 2e-16 ***
## dur5-9       0.99765    0.05275  18.912  < 2e-16 ***
## ---
## Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
## 
## (Dispersion parameter for poisson family taken to be 1)
## 
##     Null deviance: 3731.525  on 69  degrees of freedom
## Residual deviance:   70.653  on 59  degrees of freedom
## AIC: Inf
## 
## Number of Fisher Scoring iterations: 4

(exp (coef (fit))) #=> 可见随着婚龄的增长，期望的育子数将相应增长；教育程度越高，期望育子数越低；农村预期育子数比城市高等。
## (Intercept)    educnone    educsec+   educupper     resSuva    resurban 
##   1.0586073   0.9771840   0.7170105   0.8827213   0.8596609   0.9617909 
##    dur10-14    dur15-19    dur20-24    dur25-29      dur5-9 
##   3.9374452   5.0240232   5.9624936   7.2195649   2.7119024 

