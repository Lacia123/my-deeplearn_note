import torch
import torch.nn as nn
# 在开始前了解一些基本的pytorch方法
class begin():

    def __init__(self) -> None:
        
        # 开方求倒数
        t = torch.rsqrt(torch.tensor(4.0))
        print(t)
        pass

        # 创建一个全1张量
        t2 = torch.ones(3,4)
        print(t2)

# 实现RMSnorm
# RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
# 思路 
# 1、我们需要创建一个RMSnorm类，那么就需要继承nn.Module
# 2、 需要包含以下部分 1、_init_方法 _norm 主体 且 nn.Module强制要求在类中实现forward方法





class RMSNorm(nn.Module):
    # eps用避免除零，这里我们设置为1e-5
    def __init__(self, dim:int,eps:float=1e-5) -> None:
        super().__init__()
        self.dim=dim
        self.eps=eps
        #创建一个可优化的权重张量  在公式中写为weight
        self.weight= nn.Parameter(torch.ones(dim))

    #实现公式主体
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    
    def forward(self,x):
        # x.float() 先升到 float32 再做平方和均方根，避免 float16 把较大的数平方成 inf
        # 和 weight 相乘后再 type_as(x)，把结果变回 x 原来的精度
        # 若停在 float32，接到 float16 的下一层会因类型不一致报错
        return (self.weight*self._norm(x.float())).type_as(x) 


    