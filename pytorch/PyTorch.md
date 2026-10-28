# PyTorch

## nn.Parameter

`nn.Parameter` 是一种特殊的张量，用来表示模型里要学习的参数。把它赋给 `nn.Module` 的属性后，PyTorch 会自动登记它：能在 `parameters()` 里找到，会写入 `state_dict()`，优化器也会更新它。

RMSNorm 里的缩放 $\gamma$ 就是这样创建的：

```python
self.weight = nn.Parameter(torch.ones(dim))
```

`torch.ones(dim)` 得到长度为 `dim` 的全 1 向量。`nn.Parameter` 再把它标成可训练参数。默认 `requires_grad=True`，所以只要 `weight` 参与了前向计算，`loss.backward()` 就会算损失对它的偏导，存在 `weight.grad` 里，形状仍是 `(dim,)`。

普通张量默认不求梯度。`self.dim`、`self.eps` 这种 Python 数字也不在计算图里，没有梯度。

| 写法 | 是否进 `parameters()` | 是否默认求梯度 |
| --- | --- | --- |
| `torch.ones(dim)` | 否 | 否 |
| `nn.Parameter(torch.ones(dim))` | 是 | 是 |
| `self.eps = 1e-5` | 否 | 否 |

一次 `backward()` 会给本次前向里用到的所有可训练参数算出 $\partial L / \partial \theta$。没有用到的参数，`.grad` 仍是 `None`。

## torch.rsqrt

`torch.rsqrt(input)` 计算平方根的倒数，也就是 $1/\sqrt{x}$。`rsqrt(4)` 的结果是 `0.5`。

RMSNorm 用它把除法合成一步：

$$\frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}} = x \cdot \mathrm{rsqrt}(\mathrm{mean}(x^2)+\epsilon)$$

对应代码：

```python
torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

`x.pow(2)` 是逐元素平方，`mean(-1)` 在最后一维求平均，`rsqrt` 再对这个标量尺度取平方根倒数。

## mean 的 keepdim

`keepdim` 是 `mean` 的参数，不是 `rsqrt` 的参数。它决定求平均之后，被消掉的那一维还留不留。

设 `x` 的形状是 `(2, 3, 4)`，最后一维长度 4 就是公式里的 $d$：

| 写法 | 形状 | 含义 |
| --- | --- | --- |
| `x.pow(2).mean(-1)` | `(2, 3)` | 最后一维被去掉，每个 token 只剩一个数 |
| `x.pow(2).mean(-1, keepdim=True)` | `(2, 3, 1)` | 最后一维保留成长度 1 |

接下来要和原来的 `x` 相乘。PyTorch 从右边对齐形状：

- `(2, 3, 1)` 和 `(2, 3, 4)`：右边的 `1` 可以广播成 `4`，同一 token 的 4 个特征共用这一个尺度。
- `(2, 3)` 和 `(2, 3, 4)`：右边是 `3` 对 `4`，对不齐，乘法会报错。

所以 `keepdim=True` 是为了把「每个 token 一个尺度」留在最后一维，再广播回每个特征。
