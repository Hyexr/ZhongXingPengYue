# ZhongXingPengYue
13th ZhongXingPengYue
自智网络
阶段一
题目大意是求解Ax=b
A是m*n的大型稀疏矩阵，且可能不是满秩的
x需要是正整数，并且满足约束1<=x<=k

阶段二
在阶段一的基础上，b有一些扰动，即Ax+e=b
e是一些比较小的非负整数组成的向量

思路
感觉就是求且优化问题
min ||Ax-b||^2
s.t. 1<=x<=k

pyhton可以用numpy和scipy
所以先用scipy的sparse将A转为稀疏阵
lsqr来解无约束的情况，看得到x'是否满足约束
不满足就用l-bfgs-b来解有约束的情况

这种方法得到的x'不是int 所以在四舍五入得到int 但是到float最优不代表int也最优
同时阶段二 出现的扰动e也无法解决
得分
阶段一 98.58
阶段二 97.21
总分 97.62
