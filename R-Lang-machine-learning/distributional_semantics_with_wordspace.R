## Distributional Semantics 分布语义 in R with the wordspace Package ==> by Stefan Evert

(library (wordspace))
## 载入需要的程辑包：Matrix
## [1] "wordspace" "Matrix"    "stats"     "graphics"  "grDevices" "utils"    
## [7] "datasets"  "methods"   "base"     

((subset (DSM_VerbNounTriples_BNC, mode=="written")) -> Triples)

(str (Triples))
## 'data.frame':	236043 obs. of  5 variables:
##  $ noun: chr  "aa" "aa" "aa" "abandonment" ...
##  $ rel : chr  "subj" "subj" "subj" "subj" ...
##  $ verb: chr  "be" "have" "say" "be" ...
##  $ f   : num  7 5 12 14 45 13 6 23 5 7 ...
##  $ mode: Factor w/ 2 levels "spoken","written": 2 2 2 2 2 2 2 2 2 2 ...
## 

((dsm (target=Triples$noun, feature=Triples$verb, score=Triples$f, raw.freq=TRUE, sort=TRUE)) -> VObj)
## Distributional Semantic Model with 10940 rows x 3149 columns
## * raw co-occurrence matrix M available
##   - sparse matrix with 199.2k / 34.5M nonzero entries (fill rate = 0.58%)
##   - in canonical format
##   - known to be non-negative
##   - sample size of underlying corpus: 5010.1k tokens
## 

(dim (VObj)) #=> [1] 10940  3149

((subset (VObj, nnzero >= 3, nnzero >= 3, recursive=TRUE)) -> VObj)

(dim (VObj)) #=> [1] 6087 2139

((dsm.score (VObj, score="simple-ll", transform="log", normalize=TRUE)) -> VObj)

(dim (VObj)) #=>  [1] 6087 2139

## (class (VObj300)) => [1] "matrix"
((dsm.projection (VObj, method="rsvd", n=300, oversampling=4)) -> VObj300)
##                         rsvd1         rsvd2         rsvd3         rsvd4
## aa               -0.401869162 -5.715081e-01 -8.639994e-04  5.568812e-02
## abandonment      -0.164028349  7.388197e-02 -7.621238e-02 -6.218919e-02
## abbey            -0.501348714 -1.529309e-01  1.568900e-01 -3.602057e-02
## abbot            -0.556840969 -3.608457e-01  4.297981e-02 -1.349133e-02
## ability          -0.136815703  9.964295e-02 -1.508660e-01  1.483469e-01
## abnormality      -0.322822789  9.148822e-02  2.325598e-02  5.194156e-02

(dim (VObj300)) #=>  [1] 6087  300

(pair.distances ("books", "paper", VObj300, method="cosine", convert=FALSE))
## books/paper 
##        -Inf 
## 

(nearest.neighbours (VObj300, "book", n=15))
##     paper   article      poem     works  magazine     novel      text     guide 
##  45.07368  51.91946  53.48004  53.91710  53.94898  54.40499  55.13917  55.26999 
## newspaper  document      item     essay   leaflet    letter     diary 
##  55.51481  55.62563  56.28417  56.29498  56.49149  58.04203  58.11105 
## 

## 相似系数
(eval.similarity.correlation (RG65, VObj300, format="HW"))
##            rho    p.value missing         r   r.lower   r.upper
## RG65 0.3076154 0.01267694      20 0.3741373 0.1433161 0.5663555
## 

## correlation 相关性 with RG65 ratings =>>>>
## distributional model 分布模型 => distributional semantic model: 分布语义模型
## (plot (eval.similarity.correlation (RG65, VObj300, format="HW", details=TRUE))) #=> similarity_correlation.png


((nearest.neighbours (VObj300, "book", n=15)) -> nn) #=> 和书的邻居距离,只看前15个
##     paper   article      poem     works  magazine     novel      text     guide 
##  45.07368  51.91946  53.48004  53.91710  53.94898  54.40499  55.13917  55.26999 
## newspaper  document      item     essay   leaflet    letter     diary 
##  55.51481  55.62563  56.28417  56.29498  56.49149  58.04203  58.11105 
## 

## 向量拼接直接: (c ("book", (names (nn))))
((c ("book", (names (nn)))) -> nn.terms)
##  [1] "book"      "paper"     "article"   "poem"      "works"     "magazine" 
##  [7] "novel"     "text"      "guide"     "newspaper" "document"  "item"     
## [13] "essay"     "leaflet"   "letter"    "diary"    
## 

## nn.dist是个二维矩阵
((dist.matrix (VObj300, terms=nn.terms, method="cosine")) -> nn.dist)
##               book    paper  article     poem    works magazine    novel
## book       0.00000 45.07368 51.91946 53.48004 53.91710 53.94898 54.40499
## paper     45.07368  0.00000 49.58058 59.41401 63.39080 58.39195 59.63905
## article   51.91946 49.58058  0.00000 50.85024 63.56611 63.34272 56.34124
## poem      53.48004 59.41401 50.85024  0.00000 66.11456 64.23977 39.68612
## works     53.91710 63.39080 63.56611 66.11456  0.00000 64.21008 65.37230
## 

(library (MASS))
##  [1] "MASS"      "wordspace" "Matrix"    "stats"     "graphics"  "grDevices"
##  [7] "utils"     "datasets"  "methods"   "base"     
## 

((isoMDS (nn.dist, p=2)) -> mds)
## initial  value 31.571861 
## iter   5 value 27.057916
## final  value 23.256689 
## converged
## $points
##                 [,1]        [,2]
## book       -2.887478  -1.8723470
## paper     -11.206053  -6.0067030
## article     1.634778  13.8673193
## poem       17.272968  13.0067871
## works     -29.894021  11.5272668
## magazine    0.821007 -28.8959227
## novel      25.346276  -0.4610961
## text       -6.607172  22.6705171
## guide      13.584938 -10.0553809
## newspaper   6.293079 -31.0543444
## document  -14.240176   9.2152227
## item      -45.530885  -3.6561258
## essay      16.854021  14.9566610
## leaflet     4.473684 -16.7817961
## letter    -15.092965   1.4542945
## diary      39.177998  12.0856475
## 
## $stress
## [1] 23.25669
## 

## plot是画板加画图
## (plot (mds$points, pch=20, col="red"))
## 更新图画的内容
## (text (mds$points, labels=nn.terms, pos=3)) #=>> 
