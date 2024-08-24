(library ("arules"))

((read.transactions ("groceries.csv")) -> groceries)

(summary (groceries)) # ...

(inspect (groceries [1:5]))
#  [1] {bread,margarine,ready,
#       citrus,               
#       fruit,semi-finished,  
#       soups}                
#  [2] {fruit,yogurt,coffee,  
#       tropical}             
#  [3] {milk,                 
#       whole}                
#  [4] {,meat,                
#       cheese,               
#       fruit,yogurt,cream,   
#       pip,                  
#       spreads}              
#  [5] {bakery,               
#       life,                 
#       milk,condensed,       
#       milk,long,            
#       other,                
#       product,              
#       vegetables,whole}     
#  

(itemFrequency (groceries [, 1:3]))
#       ,baking     ,bottled       ,brown 
#  0.0001016777 0.0016268429 0.0020335536 

(itemFrequencyPlot (groceries, support=0.04))
## 大于0.04的支持度的数据有8种商品

(itemFrequencyPlot (groceries, topN=20))
## 前二十名销量商品: 0.7,  1.9,  3.1,  4.3,  5.5,  6.7,  7.9,  9.1, 10.3, 11.5, 12.7, 13.9, 15.1, 16.3, 17.5, 18.7, 19.9, 21.1, 22.3, 23.5

#### 绘制稀疏矩阵没有数据... ?
## (image (groceries [1:160]))

## (image (sample (groceries, 100)))

#### 符合条件的关联规则共36个
((apriori (groceries, parameter=(list (support=0.006, confidence=0.25, minlen=2)))) -> groceryrules)
#  Parameter specification:
#   confidence minval smax arem  aval originalSupport maxtime support minlen
#         0.25    0.1    1 none FALSE            TRUE       5   0.006      2
#   maxlen target   ext
#       10  rules FALSE
#  
#  Algorithmic control:
#   filter tree heap memopt load sort verbose
#      0.1 TRUE TRUE  FALSE TRUE    2    TRUE
#  
#  Absolute minimum support count: 59 
#  
#  set item appearances ...[0 item(s)] done [0.00s].
#  set transactions ...[6928 item(s), 9835 transaction(s)] done [0.00s].
#  sorting and recoding items ... [77 item(s)] done [0.00s].
#  creating transaction tree ... done [0.00s].
#  checking subsets of size 1 2 3 done [0.00s].
#  writing ... [36 rule(s)] done [0.00s].
#  creating S4 object  ... done [0.00s].
#  set of 36 rules 
#  

(summary (groceryrules))
#  rule length distribution (lhs + rhs):sizes
#   2  3 
#  28  8 
#  
#     Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#    2.000   2.000   2.000   2.222   2.000   3.000 

## pprint打印出来看inspect: 买完牛奶和奶油买奶酪
(inspect (groceryrules [1:3]))
#                  lhs         rhs     support confidence      lift count
#  [1] {beer,shopping} =>   {bags} 0.006914082  1.0000000 10.128733    68
#  [2]    {milk,cream} => {cheese} 0.006914082  1.0000000 21.662996    68
#  [3]          {milk} =>  {whole} 0.012709710  0.5530973  7.586768   125
#  
