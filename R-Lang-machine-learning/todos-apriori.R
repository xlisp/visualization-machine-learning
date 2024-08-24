(library ("arules"))

((read.transactions ("todo-itemsets.csv")) -> groceries)

(summary (groceries))
# most frequent items:
#       1   ,=,>,       2   ①,①    ,的, (Other) 
#      22      12      10       9       4    2082 


(inspect (groceries [1:5]))
# [1] {,用旧,的,华为,手机,作为,微信,爬虫,(,r,,,clojure,,,haskell,),,,提取,兴趣,点来,,,通知,消息,给,jimw,-,clj,      
#      摆脱,微信,:,连接,微信,爬虫,,,每天,的,消息,,,自动,提取,兴趣,点,,,}                                            
# [2] {①,①}                                                                                                       
# [3] {字体大小,改了}                                                                                               
# [4] {一个,屏幕,展现,多一点,内容}                                                                                  
# [5] {很困,的,时候,，,才,是,极限,的,时候,，,没有,精力,胡思乱想,，,反而,很,精神,的,时候,，,想来,些,浪费时间,的,刺激}
# 

(itemFrequency (groceries [, 1:20]))


(itemFrequencyPlot (groceries, support=0.002))
## 大于0.002的支持度的todo有7种

(itemFrequencyPlot (groceries, topN=20))

#### 绘制稀疏矩阵没有数据... ?
## (image (groceries [1:100]))

## (image (sample (groceries, 100)))

#### 符合条件的关联规则共36个
((apriori (groceries, parameter=(list (support=0.001, confidence=0.25, minlen=1)))) -> groceryrules)

(inspect (groceryrules))
#     lhs                                                                   rhs                                                                    support confidence lift count
# [1] {,机器,学习,以,特征,矩阵,为,核心,，,lisp,以,高阶,递归函数,为,核心} => {对比,学习,:,}                                                     0.001096491          1  912     2
# [2] {对比,学习,:,}                                                     => {,机器,学习,以,特征,矩阵,为,核心,，,lisp,以,高阶,递归函数,为,核心} 0.001096491          1  912     2
# 

(inspect (apriori (groceries, parameter=(list (support=0.001, confidence=0.001, minlen=1)))))
## 共有28个关联,基本上都是空对某个itemset的 => 只有`support=0.001, confidence=0.25`两个是有关联规则的
#      lhs                                                                   rhs                                                                         support  confidence lift count
# [1]  {}                                                                 => {用时,20,分钟}                                                          0.001096491 0.001096491    1     2
# [2]  {}                                                                 => {通过,id,来,操作}                                                       0.001096491 0.001096491    1     2
# [3]  {}                                                                 => {*,机器,学习,影响,整个,生活}                                            0.001096491 0.001096491    1     2
# 
