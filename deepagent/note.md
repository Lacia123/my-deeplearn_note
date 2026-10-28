## 字典式子智能体

**概念。** 用一个 Python dict 描述子智能体，交给 `create_deep_agent(..., subagents=[...])`。示例里的固定骨架是：

```python
{
    "name": "...",           # 必填：唯一 ID
    "description": "...",    # 必填：给主智能体看，决定何时调用
    "system_prompt": "...",  # 必填：子智能体自己的角色/行为
    "tools": [],             # 可选：该子智能体可用的工具列表
}
```

### 各字段含义

| 字段 | 作用 |
| --- | --- |
| `name` | 子智能体唯一标识。主智能体调内置 `task` 时，`subagent_type` 会等于这个值（如 `weather_helper`） |
| `description` | 路由说明书，主智能体主要靠它判断“该不该派这个助手” |
| `system_prompt` | 子智能体自己的系统提示，只约束它怎么回答，不给主智能体看做路由依据 |
| `tools` | 该子智能体专属工具；示例里是 `[]`，表示只靠 LLM 回答，不挂外部工具 |

未写 `model` 时，一般继承主智能体的模型。

### 和主智能体怎么接上

```python
main_agent = create_deep_agent(
    model=llm,
    tools=[],
    subagents=[weather_agent, math_agent, translate_agent],
)
```

主智能体看到的是各子智能体的 `name` + `description`；真正干活时用内置 `task` 工具，按 `subagent_type=name` 分派。

### 写的时候要注意

- `name` 要唯一、稳定（英文标识即可），后面流式日志里靠它辨认。
- `description` 要写清触发条件（“什么时候调用”），比写“我是谁”更重要。
- `system_prompt` 写执行规则（怎么答、输出格式）；和 `description` 分工不同。

### 不能嵌套

字典式子智能体的字段里没有 `subagents`。`weather_helper`、`math_helper`、`translate_helper` 是主智能体下面的并列助手，不能再在某个 dict 里面放下一层子智能体。

```python
weather_agent = {
    "name": "weather_helper",
    "description": "...",
    "system_prompt": "...",
    "tools": [],
    "subagents": [another_agent],  # 无效：dict 规格不接收这一层
}
```

要多个助手时，把它们都写进主智能体的 `subagents=[...]`。某个 dict 自己不能再按 `name` 往下分派。

## 异步流式执行

**写法。** 模型调用、工具调用、子智能体执行都在等网络 I/O。同步写法要等一个任务结束才启动下一个；异步写法让多个请求一起推进。

和同步版相比，消费流式结果只改两处：`stream` 换成 `astream`，`for` 换成 `async for`。解析 `chunk` 的逻辑不变。

```python
async def test_stream(query):
    stream = main_agent.astream({"messages": [{"role": "user", "content": query}]})
    async for chunk in stream:
        ...
```

### 协程不会立刻执行

调用 `async def` 函数只得到协程对象，函数体还没跑。

```python
task1 = test_stream("北京今天的天气怎么样？")
task2 = test_stream("请将'你是最棒的'翻译成英文。")
print(type(task1))  # <class 'coroutine'>
```

| 写法 | 实际做了什么 |
| --- | --- |
| `test_stream(query)` | 创建协程，不执行 |
| `await test_stream(query)` | 在已有事件循环里执行并等待它结束 |
| `asyncio.run(test_stream(query))` | 新建事件循环、执行、结束后关闭循环 |

`await` 只能写在 `async def` 里面。`asyncio.run` 只放在程序入口，用来启动整个异步流程；已经在事件循环里时不要再套一层 `asyncio.run`。

### 用 gather 并发

`asyncio.gather` 接收多个协程，交给同一个事件循环调度，并等到它们全部结束。谁先拿到一段流式结果，谁就先打印。

```python
async def batch_run():
    task1 = test_stream("北京今天的天气怎么样？")
    task2 = test_stream("请将'你是最棒的'翻译成英文。")
    await asyncio.gather(task1, task2)

if __name__ == "__main__":
    asyncio.run(batch_run())
```

`gather` 是并发等待，不是多进程。某个任务停在网络 I/O 时，事件循环会去推进另一个任务。

## 注册子智能体

**机制。** 官方文档里，`subagents` 放的是 `SubAgent` 字典或 `CompiledSubAgent`。多数场景用字典；复杂工作流用 `CompiledSubAgent`。同步自定义子智能体只有这两种规格，没有第三种完全不同的注册 API。差别在字典字段配多配少，以及 `mode`、`skills` 这类选项。

dict 是让框架造小兵；`CompiledSubAgent` 是把现成部队编进编制。

文档：[Subagents](https://docs.langchain.com/oss/python/deepagents/subagent)。

### 1. `SubAgent` 字典

**类型**（`TypedDict`）。框架按说明书现建子 Agent。示例 3、4 用的就是这种。

| 字段 | 作用 |
| --- | --- |
| `name` / `description` | 必填，主智能体靠它们路由 |
| `system_prompt` | `isolated` 模式必填，约束子智能体自己怎么做 |
| `tools` / `model` | 可选。不写 `tools` 时继承主智能体的工具 |
| `middleware` / `interrupt_on` / `skills` / `permissions` / `response_format` | 可选增强 |
| `mode` | `"isolated"`（默认）或 `"fork"` |

### 2. `CompiledSubAgent`

**类型**（`TypedDict`）。它和字典式平级，都放进 `create_deep_agent(..., subagents=[...])`。差别是谁来建执行体：字典由框架内部调用 `create_agent` 现建；`CompiledSubAgent` 使用你已经建好的 `runnable`。

`research_retriever_agent = create_agent(...)` 得到的是一个 Agent，不能放进 `subagents`，也不能被主智能体直接调用。要先包一层：

```python
CompiledSubAgent(
    name="research_retriever",
    description="...",
    runnable=research_retriever_agent,
)
```

主智能体仍然只看见 `name` 和 `description`，通过内置 `task` 按 `subagent_type` 分派。框架再去跑 `runnable`。`runnable` 的状态里要有 `messages`，结果才能回传。

| 字段 | 作用 |
| --- | --- |
| `name` / `description` | 必填。路由用，和字典式同一职责 |
| `runnable` | 必填。`compile` 后的图，或 `create_agent()` 的结果 |
| `mode` | 可选：`isolated` / `fork` |

官方写明 `runnable` 常见两种来源：LangChain 的 `create_agent()`，或自定义 LangGraph 图。来源可以不同，注册入口仍然是 `CompiledSubAgent`。

| 场景 | 例子 |
| --- | --- |
| 自定义 LangGraph 工作流 | 示例 6：规划流水线、条件分支 |
| LangChain `create_agent` | 示例 7：带工具的检索 Agent |
| 已有业务子图 | 审批流、ETL、多步校验 |
| 复用旧项目 | 以前写好的图或 Agent，不想改成 dict 重写 |
| 需要复杂控制 | 循环、人工确认、多节点，dict 表达不了 |

### 3. 默认 `general-purpose`

**默认实例。** 不配也会自动带一个通用子智能体，里面有文件系统等能力。可以用同名配置覆盖，或用 profile 关掉。

### 4. `AsyncSubAgent`

**类型**（`TypedDict`）。给远程、后台任务用，字段是 `graph_id`、`url` 等。它走可监控进度的长任务，不是 `subagents=[字典 / CompiledSubAgent]` 这套同步 `task`。示例 3–7 的同步注册不包含这一类。

## 创建 deep agent

**函数** `create_deep_agent`。调用它会返回一个编译好的图。模型必须支持工具调用。不额外配置时，自带这些工具：`write_todos`、文件操作（`ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`）、`execute`、`task`。`execute` 只有后端实现了沙箱协议才能真正跑命令，否则会返回错误。

```python
main_agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt="...",
    subagents=[weather_agent],
)
```

| 参数 | 作用 |
| --- | --- |
| `model` | 用哪个模型。可以是 `provider:model` 字符串，或已经初始化的聊天模型。不传时默认 `claude-sonnet-4-6` |
| `tools` | 额外工具，会和上面那套内置工具合并，不是替换 |
| `system_prompt` | 放在系统提示最前面，后面仍会接上 SDK 自带的 deep agent 提示 |
| `subagents` | 子智能体列表。三项都可以混放：字典 `SubAgent`、`CompiledSubAgent`、`AsyncSubAgent` |
| `middleware` | 插在基础中间件之后、收尾中间件之前。基础栈里已有待办、文件系统、摘要、子智能体等 |
| `skills` | 技能目录路径，如 `["/skills/user/"]`。用 POSIX 斜杠，相对后端根目录；同名技能以后写的为准 |
| `memory` | 启动时读入的记忆文件，如 `["/memory/AGENTS.md"]`，内容会补进系统提示 |
| `permissions` | 主智能体和子智能体的文件权限规则，按声明顺序匹配，先中先赢。子智能体写了自己的 `permissions` 就整份替换，不再继承 |
| `backend` | 文件存储和执行后端。要跑 `execute`，后端须实现沙箱协议 |
| `interrupt_on` | 哪些工具调用前要停下来等人确认，如 `{"edit_file": True}`。主智能体总会用这份配置；字典子智能体默认继承，自己写了就覆盖；`CompiledSubAgent` 和远程子智能体不继承 |
| `response_format` | 主智能体的结构化输出格式 |
| `context_schema` | 一次运行内不变的上下文类型，传给底层 `create_agent` |
| `checkpointer` | 在多次运行之间保存状态 |
| `store` | 持久存储。后端用 `StoreBackend` 时必须提供 |
| `debug` | 是否打开调试 |
| `name` | 这个 agent 的名字 |
| `cache` | 传给底层 `create_agent` 的缓存 |

没配名为 `general-purpose` 的子智能体时，会自动加一个同步的通用子智能体。同步子智能体一个都没有、并且 profile 关掉了这个默认项时，主智能体才不会暴露 `task` 工具。

## InMemoryStore

**类** `langgraph.store.memory.InMemoryStore`。实例是放在进程内存里的键值存储。数据随进程退出就没了，适合本地试跑。要跨进程保留，换数据库后端，例如 `PostgresStore`。`create_deep_agent` 的 `store` 可以传它；后端用 `StoreBackend` 时必须提供。

一条记录由三部分定位：`namespace` 是字符串元组，用来分组；`key` 是组内的名字；`value` 必须是字典。

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
store.put(("users", "123"), "prefs", {"theme": "dark"})
item = store.get(("users", "123"), "prefs")
store.delete(("users", "123"), "prefs")
```

| 方法 | 作用 |
| --- | --- |
| `put(namespace, key, value)` | 写入或覆盖。`value` 是字典 |
| `get(namespace, key)` | 按键读取，没有则返回 `None` |
| `delete(namespace, key)` | 删掉这一条 |
| `search(namespace_prefix, query=..., filter=..., limit=10)` | 在某个命名空间前缀下查找 |
| `list_namespaces()` | 列出已有的命名空间 |

每个方法都有对应的异步版本，如 `aput`、`aget`。

默认不做语义搜索。构造时传入 `index` 才会按向量相似度查；不传的话，`put` 上的 `index` 参数不起作用。

```python
store = InMemoryStore(index={
    "dims": 1536,
    "embed": init_embeddings("openai:text-embedding-3-small"),
    "fields": ["text"],
})
store.put(("docs",), "doc1", {"text": "Python tutorial"})
results = store.search(("docs",), query="python programming")
```

## FilesystemBackend

**类** `deepagents.backends.FilesystemBackend`。实现文件后端协议，传给 `create_deep_agent` 的 `backend`。它按真实磁盘路径读写文本，改动会直接落盘，进程退出后还在。

文档：[Backends](https://docs.langchain.com/oss/python/deepagents/backends)。

适合本地开发和 CI。不适合 Web 服务或 HTTP API，那些场景用 `StateBackend`、`StoreBackend` 或沙箱后端。它能读到权限范围内的任何文件，包括 `.env` 和密钥；写过的内容不会自动撤回。敏感操作应配 `interrupt_on` 等人确认。这个类本身不提供 `execute`，跑命令要用带沙箱或本地 shell 的后端。

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    model=llm,
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
)
```

| 参数 | 作用 |
| --- | --- |
| `root_dir` | 文件操作的根目录。不传则用当前工作目录。相对路径会被解析成绝对路径 |
| `virtual_mode` | `True` 时，路径都锚在 `root_dir` 下，并拦住 `..`、`~` 和跑出根目录的绝对路径。这只是路径限制，不是进程沙箱。`False` 时绝对路径照原样使用，`..` 也能跳出 `root_dir`。当前默认是 `False`，0.6 起必须显式传入 |
| `max_file_size_mb` | `grep` 等搜索时跳过更大的文件。默认 10 MB |

常用操作有 `ls`、`read`、`write`、`edit`、`glob`、`grep`，以及 `upload_files`、`download_files`。每个都有异步版本，如 `aread`、`awrite`。

单独使用时，智能体内部产生的大工具结果、对话历史也会写进这块真实目录。官方建议用 `CompositeBackend` 把项目目录路由到 `FilesystemBackend`，内部路径仍放在 `StateBackend`。`skills` 若已经在 `root_dir` 下，配合 `virtual_mode=True` 可以直接从磁盘加载。
