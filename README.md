### R函数式的列表(Lisp表达方式)

* lambda
```r
# define
(function (y) (function (x) ('+' (x, y))))
# call
((function (x) x) (1)) #=> [1] 1
```
* if
```r
('if' (0, ('==' (1, 1)), ('==' (2, 1)))) #=> [1] FALSE
```
* plot
```r
('plot' (('rnorm' (10)), ('rnorm' (10))))
# 加了额外的参数
('plot' (('rnorm' (10)), ('rnorm' (10)), ('=' (type, 'b'))))
```
