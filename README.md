### R函数式的列表(Lisp表达方式)
##### `C-x C-e` 执行R的S表达式
* `el-get-install ESS `
* `C-c C-k` 打开R的Repl 
* `C-x C-e` fun r lisp!
```emacs-lisp
(defun ess-eval-sexp (vis)
  (interactive "P")
  (save-excursion
    (backward-sexp)
    (let ((end (point)))
      (forward-sexp)
      (ess-eval-region (point) end vis "Eval sexp"))))
      
(add-hook 'ess-mode-hook (lambda () (define-key global-map (kbd "C-x C-e") 'ess-eval-sexp) ))
```

##### lambda
```r
# define
(function (y) (function (x) ('+' (x, y))))
# call
((function (x) x) (1)) #=> [1] 1
```
##### if
```r
('if' (0, ('==' (1, 1)), ('==' (2, 1)))) #=> [1] FALSE
```
##### plot
```r
('plot' (('rnorm' (10)), ('rnorm' (10))))
# 加了额外的参数
('plot' (('rnorm' (10)), ('rnorm' (10)), ('=' (type, 'b'))))
```
##### Reduce
```r
(Reduce ('*', 1:10))
```
##### Filter
```r
((function (x) ('if' (('%%' (x, 2)), x, 0))) (2)) #=> [1] 0
# call
(Filter ((function (x) ('if' (('%%' (x, 2)), x, 0))), 1:10)) #=>  [1] 1 3 5 7 9
```
##### Map
```r
(Map ((function (x) ('+' (x, 100))), 1:3))
# =>
[[1]]
[1] 101
[[2]]
[1] 102
[[3]]
[1] 103
```
##### vector
```r
# 如果本来是前缀的表达方式的函数,引号'c'可以省略,function除外必须加引号
(c (1, 1, 3)) #=> [1] 1 1 3
((c (1, 8, 3)) [2]) #=> [1] 8
('=' (defvar, (c ("A", "B", "C")))) #=> [1] "A" "B" "C"
```
##### factor
```r
(factor ((c ("1", "1", "3")), ('=' (levels, (c ("A", "B", "C"))))))
#=>
[1] <NA> <NA> <NA>
Levels: A B C
```
##### list
```r
(list (11, "aa", FALSE))
#=>
[[1]]
[1] 11
[[2]]
[1] "aa"
[[3]]
[1] FALSE
```
##### data.frame (函数内赋值参数用: x=123)
```r
('=' (pt_data,
  (data.frame (
    ID=(c (11,12,13)),
    Name=(c ("Devin","Edward","Wenli")),
    Gender=(c ("M","M","F")),
    Birthdate=(c ("1984-12-29","1983-5-6","1986-8-8"))))))
#=>
  ID   Name Gender  Birthdate
1 11  Devin      M 1984-12-29
2 12 Edward      M   1983-5-6
3 13  Wenli      F   1986-8-8

## get:
(pt_data [1, 2]) #=> 第一行,第二列
[1] Devin
Levels: Devin Edward Wenli

(pt_data [,3]) #=> 只是第三列
[1] M M F
Levels: F M
```
##### matrix (函数内赋值参数用: x=123)
```r
(matrix ((c (1, 2, 1, 3, 5, 8)), nrow=2)) 
#=>  2行->3列
     [,1] [,2] [,3]
[1,]    1    1    5
[2,]    2    3    8

(matrix ((c (1, 2, 1, 3, 5, 8)), ncol=2))
#=>
     [,1] [,2]
[1,]    1    3
[2,]    2    5
[3,]    1    8

(matrix ((c (1, 2, 4, 3)), ncol=1))
#=> 单列矩阵
     [,1]
[1,]    1
[2,]    2
[3,]    4
[4,]    3

(matrix ((c (1, 2, 4, 3)), nrow=1))
#=> 单行矩阵
     [,1] [,2] [,3] [,4]
[1,]    1    2    4    3

```
##### csv 表格数据文件
```r
(write.csv (pt_data, ('=' (file, "my-data-frame.csv"))))
# cat my-data-frame.csv #=>
"","ID","Name","Gender","Birthdate"
"1",11,"Devin","M","1984-12-29"
"2",12,"Edward","M","1983-5-6"
"3",13,"Wenli","F","1986-8-8"

(read.csv ("my-data-frame.csv"))
#=>
  X ID   Name Gender  Birthdate
1 1 11  Devin      M 1984-12-29
2 2 12 Edward      M   1983-5-6
3 3 13  Wenli      F   1986-8-8
```
