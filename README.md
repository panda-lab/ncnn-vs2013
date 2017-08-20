# ncnn-vs2013
- ncnn的vs2013工程
- caffe2ncnn.bin是windows上的可执行文件，可以直接使用啦
- 修改了归一化函数，适用于不同的归一化方式(mat.cpp,line 80-83)
- 加入了我们训练的face landmark https://github.com/lsy17096535/face-landmark (更多结果请查看该项目)
- 在i5-cpu caffe上不到5ms，ncnn上要慢一点，估计是在arm上做的优化
- 归一化只在c代码中做了，arm的simd没有做，如果移植到arm上，参考Issue #2 ，修改几行代码即可
# Reference
- http://blog.csdn.net/fuwenyan/article/details/76105574
- https://github.com/guozhongluo/ncnn-vs2015-examples-demo

# Result
![](ncnn/result.jpg)
