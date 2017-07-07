
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

