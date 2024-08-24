(library ("arules"))

((read.transactions ("todo-itemsets-1.csv")) -> groceries)
# 机器,特征,bb ### 识别不了
# 机器,特征,aa
# 机器,特征,cc

(inspect (apriori (groceries, parameter=(list (support=0.001, confidence=0.001, minlen=1)))))
