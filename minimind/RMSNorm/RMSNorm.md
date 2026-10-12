# RMSNorm

RMSNorm 用在每个 token 进入注意力或前馈层之前，把最后一维的幅度拉到相近的尺度，方向基本保留。大模型里它常用来代替 LayerNorm：少做一次减均值，计算更轻，训练也更稳。

## 类型区别

三种归一化看的范围不同。

| 类型 | 统计范围 | 做了什么 |
| --- | --- | --- |
| BatchNorm | 一个 batch 里、同一通道的所有样本 | 减均值、除标准差。依赖 batch，序列模型里不常用 |
| LayerNorm | 单个 token 的全部特征 | 先减均值，再除标准差，通常还有缩放 $\gamma$ 和偏移 $\beta$ |
| RMSNorm | 单个 token 的全部特征 | 只除均方根，不减均值，一般只有缩放 $\gamma$，没有偏移 |

LayerNorm 把向量拉成「均值 0、方差 1」。RMSNorm 认为减均值收益不大，只按均方根缩放，所以更省算力。

## 公式

对最后一维长度为 $d$ 的向量 $x$：

$$\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2} = \sqrt{\mathrm{mean}(x^2)}$$

$$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}} \odot \gamma$$

$\odot$ 是逐元素相乘。代码里用 `rsqrt`（平方根的倒数）一次做完除法：

```python
def _norm(x, eps):
    # x: (..., dim)，只在最后一维上求均方
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

def forward(x, weight):
    return weight * _norm(x, eps)
```

`mean` 要 `keepdim=True`，结果才能和 $x$ 逐元素相乘。

## 参数

| 参数 | 是否学习 | 作用和目的 |
| --- | --- | --- |
| `x` | 否 | 输入，形状 `(..., dim)`。每个样本只看自己的最后一维 |
| `dim` | 否 | 最后一维长度，同时决定 `weight` 的长度，要和隐藏层宽度一致 |
| `eps` | 否 | 加在均方根里面，避免 $x$ 接近 0 时除以 0。常用 `1e-5` 或 `1e-6`，只为数值稳定 |
| `weight`（$\gamma$） | 是 | 形状 `(dim,)`，初始全 1。归一化会抹掉原有尺度，用它把每个特征的尺度重新学回来 |

`weight` 初始为 1 时，一开始等价于只做均方根缩放，训练再慢慢改各维的比例。
