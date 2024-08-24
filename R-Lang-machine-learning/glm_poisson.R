(data (warpbreaks))

(head (warpbreaks))
##   breaks wool tension
## 1     26    A       L
## 2     30    A       L
## 3     54    A       L
## 4     25    A       L
## 5     70    A       L
## 6     52    A       L

((glm (breaks ~ tension, data=warpbreaks, family="poisson")) -> rs1)
## Coefficients:
## (Intercept)     tensionM     tensionH
##      3.5943      -0.3213      -0.5185
##
## Degrees of Freedom: 53 Total (i.e. Null);  51 Residual
## Null Deviance:	    297.4
## Residual Deviance: 226.4 	AIC: 507.1
##

(summary (rs1))
## Deviance Residuals:
##     Min       1Q   Median       3Q      Max
## -4.2464  -1.6031  -0.5872   1.2813   4.9366
##
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)
## (Intercept)  3.59426    0.03907  91.988  < 2e-16 ***
## tensionM    -0.32132    0.06027  -5.332 9.73e-08 ***
## tensionH    -0.51849    0.06396  -8.107 5.21e-16 ***
## ---
## Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
##
## (Dispersion parameter for poisson family taken to be 1)
##
##     Null deviance: 297.37  on 53  degrees of freedom
## Residual deviance: 226.43  on 51  degrees of freedom
## AIC: 507.09
##
## Number of Fisher Scoring iterations: 4
