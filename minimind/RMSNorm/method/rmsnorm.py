import torch
# 在开始前了解一些基本的pytorch方法
class begin():

    def __init__(self) -> None:
        
        # 开方求倒数
        t = torch.rsqrt(torch.rensor(4.0))
        print(t)
        pass

        # 创建一个全1张量
        t2 = torch.ones(3.4)
        print(t2)

# 实现