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
* Reduce
```r
(Reduce ('*', 1:10))
```
* Filter
```r
((function (x) ('if' (('%%' (x, 2)), x, 0))) (2)) #=> [1] 0
# call
(Filter ((function (x) ('if' (('%%' (x, 2)), x, 0))), 1:10)) #=>  [1] 1 3 5 7 9
```
* Map
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
