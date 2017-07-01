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

## ########### 5. 为频繁出现的单词创建指示特征 ===>> 未完待续。。。

