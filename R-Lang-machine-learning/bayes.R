## ########## 1. 读入数据,初略判断数据特征
('<-' (sms_raw, (read.csv ("http://127.0.0.1:8003/sms_spam.csv", stringsAsFactors=FALSE))))

(class (sms_raw)) #=>  [1] "data.frame"

(str (sms_raw)) # 数据结构探索函数str: 每条短信都有两个特征: type & text
## 'data.frame':	5574 obs. of  2 variables:
##  $ type: chr  "ham" "ham" "spam" "ham" ...
##  $ text: chr  "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..." "Ok
 
(sms_raw$type) #=> [5561] "ham" "spam"... 当前的变量type是个字符串的向量=> 分类向量需要转为因子将会更好, 分类垃圾短信还是有用短信

(factor (sms_raw$type)) #=> [5559] ham  ham spam

('<-' (sms_raw$type, (factor (sms_raw$type)))) #=> 修改"type是个字符串的向量"为因子, 方便table分析

(str (sms_raw$type))
##  Factor w/ 2 levels "ham","spam": 1 1 2 1 1 2 1 1 2 2 ...

(table (sms_raw$type)) #=>
##  ham spam 
## 4827  747 

## ########### 2.处理和分析文本数据,用NLP的包tm
(library (tm)) #=> NLP自然语言处理的包

(class ((VectorSource (sms_raw$text)))) # [1] "VectorSource" "SimpleSource" "Source"      

('<-' (sms_corpus, (Corpus ((VectorSource (sms_raw$text)))))) ##Corpus来创建语料库,包含5574条训练数据短信, Corpus还支持PDF和office文档
## <<SimpleCorpus>>
## Metadata:  corpus specific: 1, document level (indexed): 0
## Content:  documents: 5574

(print (sms_corpus)) #=> 打印语料库的基本信息
## <<SimpleCorpus>>
## Metadata:  corpus specific: 1, document level (indexed): 0
## Content:  documents: 5574
## <<SimpleCorpus>>
## Metadata:  corpus specific: 1, document level (indexed): 0
## Content:  documents: 5574
##  

(inspect (sms_corpus [1:3])) # 查看具体的短信内容
## <<SimpleCorpus>>
## Metadata:  corpus specific: 1, document level (indexed): 0
## Content:  documents: 3
## 
## [1] Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat           
## [2] Ok lar... Joking wif u oni...                                                                                                

(inspect ((tm_map ((sms_corpus [1:3]), removeNumbers))))
## map去掉数字以前 ==>> [3] Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.
## map去掉数字之后 ==>> [3] Free entry in  a wkly comp to win FA Cup final tkts st May .

('<-' (corpus_clean, (tm_map ((tm_map (sms_corpus, removeNumbers)), tolower))))

(inspect ((corpus_clean [1:3]))) #=> 所有的字母都变成小写了, 所有的数字都去除了

## 去除停用词,和标点符号
('<-' (corpus_clean, (tm_map ((tm_map (corpus_clean, removeWords, (stopwords ()))), removePunctuation))))
## 将空格都变成单个空格
('<-' (corpus_clean, (tm_map (corpus_clean, stripWhitespace))))

('<-' (sms_dtm, (DocumentTermMatrix (corpus_clean)))) #=> 将清洗好的语料库变成矩阵
## <<DocumentTermMatrix (documents: 5574, terms: 7917)>>
## Non-/sparse entries: 43073/44086285
## Sparsity           : 100%
## Maximal term length: 40
## Weighting          : term frequency (tf)
## 

(class (sms_dtm)) #=> [1] "DocumentTermMatrix"    "simple_triplet_matrix"

## ########### 3. 建立训练数据集合测试数据集 (同一数据源大部分数据作为训练数据, 小部分作为测试数据)
## 原始数据data.frame
('<-' (sms_raw_train, (sms_raw [1:4169, ])))
('<-' (sms_raw_test, (sms_raw [4170:5574, ])))
## 语料库矩阵数据  [1] "DocumentTermMatrix"    "simple_triplet_matrix"
('<-' (sms_dtm_train, (sms_dtm [1:4169, ])))
('<-' (sms_dtm_test, (sms_dtm [4170:5574, ])))
## 语料库
('<-' (sms_corpus_train, (corpus_clean [1:4169])))
('<-' (sms_corpus_test, (corpus_clean [4170:5574])))
## 确保训练数据和测试数据无误
(round (('*' ((prop.table (table (sms_raw_train$type))) ,100)), digits=1))
##  ham spam 
## 86.5 13.5 
(round (('*' ((prop.table (table (sms_raw_test$type))) ,100)), digits=1))
##  ham spam 
##   87   13

## ############ 4. 可视化文本数据---标签云(词云)
(library (wordcloud))
## 载入需要的程辑包：RColorBrewer
##  [1] "wordcloud"    "RColorBrewer" "tm"           "NLP"          "stats"       
##  [6] "graphics"     "grDevices"    "utils"        "datasets"     "methods"     
## [11] "base"        

## 从tm语料库sms_corpus_train里直接创建词云: 已经生成了标签云的图了, 演示数据使用如只是看垃圾的标签数据
(wordcloud (sms_corpus_train, min.freq=40, random.order=FALSE)) #=> wordcloud_01.png

## ########### 5. 为频繁出现的单词创建指示特征 ===>> 未完待续。。。在需要它的时候,分布式更新它 7-01创建 ~> 8-17回归更新
## 该向量中的单词,在矩阵sms_dtm_train中至少出现5次=>
((findFreqTerms (sms_dtm_train, 5)) -> sms_dict)
## ... ...
## [1217] "murdered"        "murderer"        "police"          "budget"         
## [1221] "happens"         "thurs"          

((DocumentTermMatrix (sms_corpus_train, (list (dictionary=sms_dict)))) -> sms_train)
## <<DocumentTermMatrix (documents: 4169, terms: 1222)>>
## Non-/sparse entries: 24047/5070471
## Sparsity           : 100%
## Maximal term length: 15
## Weighting          : term frequency (tf)

((DocumentTermMatrix (sms_corpus_test, (list (dictionary=sms_dict)))) -> sms_test)
## <<DocumentTermMatrix (documents: 1405, terms: 1222)>>
## Non-/sparse entries: 7775/1709135
## Sparsity           : 100%
## Maximal term length: 15
## Weighting          : term frequency (tf)

## (convert_counts (-1)) #=>[1] No, 量化或者是因子化
((function (x, y=(ifelse (x > 0, 1, 0)))
    (factor (y, levels=(c (0, 1)), labels=(c ("No", "Yes"))))) -> convert_counts)

((apply (sms_train, MARGIN=2, convert_counts)) -> sms_train)
((apply (sms_test, MARGIN=2, convert_counts)) -> sms_test)

## ########## 6. 基于数据训练模型

(library (e1071))
## [1] "e1071"        "wordcloud"    "RColorBrewer" "tm"           "NLP"         
## [6] "stats"        "graphics"     "grDevices"    "utils"        "datasets"    
##[11] "methods"      "base"        

## naiveBayes("训练数据的矩阵或者dataframe", "训练数据每行的分类的一个因子向量")
((naiveBayes (sms_train, sms_raw_train$type)) -> sms_classifier)

## 进行预测
## (predict (sms_classifier, "测试数据的矩阵", type="class")

## (str (sms_raw_train)) #==>>
##'data.frame':	4169 obs. of  2 variables:
## $ type: Factor w/ 2 levels "ham","spam": 1 1 2 1 1 2 1 1 2 2 ...
## $ text: chr  "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..." "Ok lar... Joking wif u oni..." "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question("| __truncated__ "U dun say so early hor... U c already then say..." ...
##
##


### sms_classifier ===>>> sms_raw_train$type
##Naive Bayes Classifier for Discrete Predictors
##
##Call:
##naiveBayes.default(x = sms_train, y = sms_raw_train$type)
##
##A-priori probabilities:
##sms_raw_train$type
##      ham      spam 
##0.8647158 0.1352842 
##
##Conditional probabilities:
##                  available
##sms_raw_train$type          No         Yes
##              ham  0.996393897 0.003606103
##              spam 0.996453901 0.003546099
##
##                  bugis
##sms_raw_train$type          No         Yes
##              ham  0.998335645 0.001664355
##              spam 1.000000000 0.000000000
##
##                  cine
##sms_raw_train$type          No         Yes
##              ham  0.998058252 0.001941748
##              spam 1.000000000 0.000000000
##
##                  crazy
##sms_raw_train$type          No         Yes
##              ham  0.997780860 0.002219140
##              spam 0.996453901 0.003546099
##
##                  got
##sms_raw_train$type          No         Yes
##              ham  0.954230236 0.045769764
##              spam 0.994680851 0.005319149
##
##                  great
##sms_raw_train$type          No         Yes
##              ham  0.981137309 0.018862691
##              spam 0.991134752 0.008865248
##
##                  point
##sms_raw_train$type          No         Yes
##              ham  0.996948682 0.003051318
##              spam 1.000000000 0.000000000
##
##                  wat
##sms_raw_train$type         No        Yes
##              ham  0.98085992 0.01914008
##              spam 1.00000000 0.00000000
##
##                  world
##sms_raw_train$type          No         Yes
##              ham  0.993065187 0.006934813
##              spam 0.998226950 0.001773050
##
##                  lar
##sms_raw_train$type          No         Yes
##              ham  0.991955617 0.008044383
##              spam 1.000000000 0.000000000
##
##                  wif
##sms_raw_train$type          No         Yes
##              ham  0.995006935 0.004993065
##              spam 1.000000000 0.000000000
##
##                  apply
##sms_raw_train$type          No         Yes
##              ham  0.999445215 0.000554785
##              spam 0.959219858 0.040780142
##
##                  comp
##sms_raw_train$type         No        Yes
##              ham  1.00000000 0.00000000
##              spam 0.98758865 0.01241135
##
##                  cup
##sms_raw_train$type           No          Yes
##              ham  0.9991678225 0.0008321775
##              spam 0.9911347518 0.0088652482
##
##                  entry
##sms_raw_train$type         No        Yes
##              ham  1.00000000 0.00000000
##              spam 0.96985816 0.03014184
##
##                  final
##sms_raw_train$type          No         Yes
##              ham  0.999445215 0.000554785
##              spam 0.975177305 0.024822695
##
##
##                  free
##sms_raw_train$type         No        Yes
##              ham  0.98834951 0.01165049
##              spam 0.77127660 0.22872340
##
##                  may
##sms_raw_train$type          No         Yes
##              ham  0.991400832 0.008599168
##              spam 0.989361702 0.010638298
##
##                  receive
##sms_raw_train$type          No         Yes
##              ham  0.998613037 0.001386963
##              spam 0.960992908 0.039007092
##
##                  text
##sms_raw_train$type         No        Yes
##              ham  0.98557559 0.01442441
##              spam 0.86347518 0.13652482
##
##                  txt
##sms_raw_train$type          No         Yes
##              ham  0.998058252 0.001941748
##              spam 0.796099291 0.203900709
##
##                  win
##sms_raw_train$type          No         Yes
##              ham  0.998335645 0.001664355
##              spam 0.913120567 0.086879433
##
##                  wkly
##sms_raw_train$type         No        Yes
##              ham  1.00000000 0.00000000
##              spam 0.98404255 0.01595745
##
##                  already
##sms_raw_train$type         No        Yes
##              ham  0.98141470 0.01858530
##              spam 0.99822695 0.00177305
##
##                  dun
##sms_raw_train$type         No        Yes
##              ham  0.99001387 0.00998613
##              spam 1.00000000 0.00000000
##
##                  early
##sms_raw_train$type          No         Yes
##              ham  0.992510402 0.007489598
##              spam 1.000000000 0.000000000
##
##
## ... ... ...

