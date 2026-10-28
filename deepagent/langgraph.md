## workflow 方法

`StateGraph(...)` 得到的 `workflow` 还只是一张画布。下面这些方法依次往上登记步骤和走法，`compile()` 之后才变成可执行的图。

`add_node` 定有哪些步骤，`set_entry_point`、`add_edge`、`add_conditional_edges` 定怎么走，`compile` 之后才能跑。`CompiledSubAgent` 不是 `workflow` 的方法，它是 DeepAgents 的包装。

### 1. `StateGraph`：创建图

```python
workflow = StateGraph[ResearchPlanState, None, ResearchPlanState, ResearchPlanState](ResearchPlanState)
```

这是一张空的状态图，还不能跑。括号里的 `ResearchPlanState` 是这张图的 state 结构。前面的 `[...]` 是更完整的泛型写法，标出输入、输出等状态类型。教学上等价于：

```python
workflow = StateGraph(ResearchPlanState)
```

此时只有画布，还没有节点和边。

### 2. `add_node`：登记步骤

```python
workflow.add_node("extract_topic", extract_topic)
workflow.add_node("quick_plan", build_quick_plan)
```

| 参数 | 含义 |
| --- | --- |
| 第 1 个 | 节点 ID（字符串），边里用这个名字跳转 |
| 第 2 个 | 节点函数：`(state) -> 要更新的字段` |

这里只登记有哪些工序，还不决定顺序。

### 3. `set_entry_point`：起点

```python
workflow.set_entry_point("extract_topic")
```

图从哪个节点开始跑。没有入口，`compile` 之后也不知道从哪开始。新版也可以写成 `workflow.add_edge(START, "extract_topic")`，意思一样。

### 4. `add_conditional_edges`：带判断的箭头

```python
workflow.add_conditional_edges(
    "extract_topic",      # 从哪出发
    route_by_depth,       # 路由函数：看 state，返回一个 key
    {
        "quick": "quick_plan",  # 返回 "quick" → 去这
        "deep": "deep_plan",    # 返回 "deep"  → 去那
    },
)
```

三件套：

- 源节点：跑完后要分支。
- 路由函数：不改业务数据，只返回字符串，或其它能拿去查表的值。
- 路径表：返回值对应下一节点名。

本例里 `route_by_depth` 返回 `state["depth"]`，再查表跳转。

### 5. `add_edge`：固定箭头

```python
workflow.add_edge("quick_plan", "finalize_plan")
workflow.add_edge("deep_plan", "finalize_plan")
workflow.add_edge("finalize_plan", END)
```

| 写法 | 含义 |
| --- | --- |
| `add_edge("A", "B")` | A 结束一定去 B |
| `add_edge(..., END)` | 去内置终点，图结束 |

这里让 `quick` 和 `deep` 汇合到 `finalize_plan`，再结束。

### 6. `compile`：蓝图变成可执行程序

```python
compiled_graph = workflow.compile()
```

`compile` 检查节点和边是否合法，生成可以 `invoke`、`stream` 的 Runnable。`workflow` 是配置对象；真正被 DeepAgents 调用的是 `compiled_graph`。

### 7. `CompiledSubAgent`：挂到 DeepAgents

这不是 `workflow` 的方法，而是 DeepAgents 的包装，让主智能体能把这张图当子智能体调用。

| 字段 | 作用 |
| --- | --- |
| `name` | 子智能体 ID；`task` 的 `subagent_type` 对应它 |
| `description` | 给主 LLM 看，决定何时调用 |
| `runnable` | 真正的执行体，这里就是 `compiled_graph` |

```text
StateGraph(State)          # 建空图
  .add_node(...)           # 加工序
  .set_entry_point(...)    # 定起点
  .add_conditional_edges   # 加分支箭头
  .add_edge(...)           # 加固定箭头
  .compile()               # 变成可跑的图
       │
       ▼
CompiledSubAgent(runnable=...)  # 挂到 DeepAgents
```
