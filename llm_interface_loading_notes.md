# RAGFlow LLM 接口封装与加载机制笔记

## 一句话总结

RAGFlow 对 LLM 的调用采用了：

> 配置驱动 + 注册表查类 + 工厂方法实例化 + 面向接口调用

也就是说，业务代码不直接依赖某个具体厂商模型，而是通过统一入口拿到模型对象，再调用统一方法。

## 1. 整体调用链路

```text
用户/租户配置模型
  ↓
生成 model_config
  ↓
TenantLLMService.model_instance(model_config)
  ↓
根据 model_type 选择注册表
  ↓
根据 llm_factory 查找具体实现类
  ↓
实例化具体模型类
  ↓
封装进 LLMBundle
  ↓
业务层通过统一方法调用
```

例如：

```text
model_type = chat
llm_factory = OpenAI
llm_name = gpt-4o
```

最终会变成某个具体 Chat 实现类实例，然后业务层通过：

```python
await chat_mdl.async_chat(...)
```

调用。

## 2. 配置驱动

模型选择不是写死在代码里的，而是由配置决定。

核心配置字段通常包括：

```python
{
    "model_type": "chat",
    "llm_factory": "OpenAI",
    "llm_name": "gpt-4o",
    "api_key": "...",
    "api_base": "..."
}
```

其中：

```text
model_type  决定模型能力类型
llm_factory 决定具体厂商
llm_name    决定具体模型名称
api_key     用于鉴权
api_base    用于请求模型服务
```

常见能力类型：

```text
chat       -> 聊天模型
embedding  -> 向量模型
rerank     -> 重排模型
image2text -> 图像理解模型
tts        -> 文字转语音
ocr        -> 文档解析/OCR
```

## 3. 注册表模式

RAGFlow 会维护多张模型注册表：

```python
ChatModel = {}
EmbeddingModel = {}
RerankModel = {}
CvModel = {}
Seq2txtModel = {}
TTSModel = {}
OcrModel = {}
```

每张表负责一种模型能力。

例如：

```python
EmbeddingModel["OpenAI"] = OpenAIEmbed
EmbeddingModel["HuggingFace"] = HuggingFaceEmbed
ChatModel["Tongyi-Qianwen"] = QWenChat
ChatModel["OpenAI"] = OpenAIChat
```

所以运行时可以通过字符串找到具体类：

```python
EmbeddingModel[model_config["llm_factory"]]
ChatModel[model_config["llm_factory"]]
```

这就是注册表模式。

## 4. 自动注册机制

注册表不是完全手写的，而是在 `rag/llm/__init__.py` 中动态扫描模型模块。

它会 import 这些模块：

```text
chat_model.py
embedding_model.py
rerank_model.py
cv_model.py
sequence2txt_model.py
tts_model.py
ocr_model.py
```

然后查找继承自 `Base` 的类，并读取类上的 `_FACTORY_NAME`。

例如：

```python
class OpenAIEmbed(Base):
    _FACTORY_NAME = "OpenAI"
```

注册后就会形成：

```python
EmbeddingModel["OpenAI"] = OpenAIEmbed
```

因此新增模型厂商时，一般只需要：

```text
1. 新增一个实现类
2. 继承对应 Base
3. 实现统一方法
4. 设置 _FACTORY_NAME
```

## 5. 工厂方法实例化

核心工厂方法是：

```python
TenantLLMService.model_instance(model_config)
```

它的职责是：

```text
根据 model_config 创建具体模型对象
```

简化逻辑如下：

```python
if model_type == "embedding":
    return EmbeddingModel[llm_factory](...)

elif model_type == "chat":
    return ChatModel[llm_factory](...)

elif model_type == "rerank":
    return RerankModel[llm_factory](...)
```

也就是：

```text
model_type 选择注册表
llm_factory 选择具体类
```

这就是典型的工厂方法。

## 6. 面向接口调用

不同模型实现类都会继承统一基类。

例如 embedding 模型基类：

```python
class Base:
    def encode(self, texts):
        raise NotImplementedError

    def encode_queries(self, text):
        raise NotImplementedError
```

具体实现类负责实现这些方法：

```python
class OpenAIEmbed(Base):
    def encode(self, texts):
        ...
```

业务层不关心具体是 OpenAI、Qwen、HuggingFace 还是 Ollama，只关心它能不能调用：

```python
encode()
chat()
similarity()
describe()
transcription()
tts()
```

## 7. LLMBundle 的作用

`LLMBundle` 是模型调用的统一门面。

它内部持有具体模型实例：

```python
self.mdl = TenantLLMService.model_instance(model_config)
```

真正调用模型的是：

```python
self.mdl
```

但业务层通常通过 `LLMBundle` 调用：

```python
llm.encode(...)
llm.async_chat(...)
llm.similarity(...)
llm.describe(...)
```

`LLMBundle` 额外负责：

```text
1. 调用具体模型实现
2. 统一接收返回值
3. 更新 token 使用量
4. 记录 Langfuse trace
5. 清理模型输出
6. 处理工具调用输出
7. 做部分输入裁剪
```

所以它可以理解为一个门面模式/适配层。

## 8. Chat 返回结果统一

不同厂商的原始返回格式不同。

例如：

```text
OpenAI: response.choices[0].message.content
Gemini: candidates / parts
Anthropic: content blocks
Qwen: output / choices / usage
```

RAGFlow 在各厂商实现类中把它们转换成统一格式：

```python
(txt, used_tokens)
```

也就是：

```text
txt         -> 模型回答文本
used_tokens -> token 消耗
```

例如：

```python
ans = response.choices[0].message.content.strip()
return ans, total_token_count_from_response(response)
```

然后 `LLMBundle.async_chat()` 接收：

```python
txt, used_tokens = await chat_partial(...)
```

再做清洗和统计，最后业务层通常只拿到：

```python
txt
```

## 9. 同步、异步、流式调用

RAGFlow 中常见几类 chat 方法：

```text
chat                  同步，一次性返回完整结果
async_chat            异步，一次性返回完整结果
chat_streamly         同步，流式返回
async_chat_streamly   异步，流式返回
```

区别：

```text
chat        会阻塞当前流程
async_chat  需要 await，适合 Web 服务并发场景
streamly    一边生成一边返回
```

## 10. 其他能力的返回约定

RAGFlow 没有把所有模型返回都包装成同一个 `LLMResponse` 对象，而是按能力类型约定格式。

常见约定：

```text
chat             -> 底层返回 (txt, used_tokens)，LLMBundle 对外返回 txt
embedding        -> (embeddings, used_tokens)
rerank           -> (scores, used_tokens)
image2text       -> 底层返回 (txt, used_tokens)，LLMBundle 对外返回 txt
speech2text      -> 底层返回 (txt, used_tokens)，LLMBundle 对外返回 txt
tts              -> bytes 流，夹带 token 统计
ocr              -> 文档解析结果
```

## 11. DB.connection_context 的作用

很多 service 方法上有：

```python
@DB.connection_context()
```

它的作用是：

```text
在方法执行前确保数据库连接可用
方法执行结束后自动释放/收尾连接
```

等价于：

```python
with DB.connection_context():
    ...
```

它不是模型工厂逻辑本身，而是数据库连接管理。

## 12. 设计模式总结

### 配置驱动

模型选择由配置决定，而不是硬编码。

```text
model_config -> 决定用哪个模型
```

### 注册表模式

用字典维护厂商名到实现类的映射。

```python
ChatModel["OpenAI"] = OpenAIChat
```

### 工厂方法

通过统一方法根据配置创建具体实例。

```python
TenantLLMService.model_instance(model_config)
```

### 面向接口编程

业务层只调用统一方法，不关心具体厂商。

```python
llm.async_chat(...)
llm.encode(...)
llm.similarity(...)
```

### 门面模式

`LLMBundle` 作为统一入口，封装 token 统计、trace、清洗等公共逻辑。

## 13. 最终理解

RAGFlow 的 LLM 调用架构可以概括为：

```text
配置决定模型
注册表找到类
工厂创建实例
LLMBundle 统一封装
业务层面向接口调用
```

更完整地说：

```text
不同厂商模型
  ↓
实现统一接口
  ↓
注册到模型注册表
  ↓
运行时根据配置选择实现类
  ↓
通过工厂方法实例化
  ↓
封装进 LLMBundle
  ↓
业务层统一调用
```

这个设计的好处是：

```text
1. 新增模型厂商成本低
2. 业务层不用感知厂商差异
3. token 统计和 trace 可以集中处理
4. 支持多租户不同模型配置
5. 支持 chat、embedding、rerank、ocr 等多种模型能力
```
