(library (Boruta))
(library (mlbench))
## [1] "mlbench"   "stats"     "graphics"  "grDevices" "utils"     "datasets"
## [7] "methods"   "base"

(data ("Ozone")) #=>  [1] "Ozone", dataframe
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
