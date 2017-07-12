
(library (bmp))

## (class (lenna)) #=>  [1] "matrix" , 将图片导入为数值矩阵, 但是是倒过来的
((read.bmp ("lena512.bmp")) -> lenna)

## 旋转图片
(((t (lenna)) [,(nrow (lenna)):1]) -> lenna)
## (image (lenna)) 查看到是橘红色的旋转后图片

## 进行SVD操作,保存到新的变量lenna.svd, 绘制方差的百分比图
((svd (scale (lenna))) -> lenna.svd)

## (plot (('/' (lenna.svd$d^2, (sum (lenna.svd$d^2)))), type="l", xlab=" Singular vector", ylab = "Variance explained"))
##=> variance_percentage.png

(length (lenna.svd$d)) #=> [1] 512

## !!! 找到能解释90%以上变量的奇异向量数据点: 90%相似度需要27个奇异向量才能达到
(min (which ('>' ((cumsum ('/' (lenna.svd$d^2, (sum (lenna.svd$d^2))))), 0.9))))
##=>  [1] 27

((function (dim,
            u=(as.matrix (lenna.svd$u[, 1:dim])),
            v=(as.matrix (lenna.svd$v[, 1:dim])),
            d=(as.matrix ((diag (lenna.svd$d)) [1:dim, 1:dim])))
    (image ('%*%' (('%*%' (u, d)), (t (v)))) ) ) -> lenna_compression)

## (lenna_compression (27))
