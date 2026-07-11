# Akashic Agent 配置指南

## 环境要求

- Python 3.12
- 推荐使用 `uv` 管理环境和依赖

安装 `uv`：

```bash
pip install uv
```

## 安装依赖

```bash
git clone https://github.com/Vannila0v0/chatbot.git
cd chatbot
uv venv
uv pip install -r requirements.txt
```

## 初始化配置

推荐使用交互式配置向导：

```bash
uv run python main.py setup
```

也可以执行非交互初始化，再手动编辑根目录的 `config.toml`：

```bash
uv run python main.py init
```

仓库中的 `config.example.toml` 是配置模板。不要把填写了真实密钥的 `config.toml` 提交到 Git。

## 配置模型

下面是一份基础配置示例：

```toml
[llm]
provider = "deepseek"

[llm.main]
model = "deepseek-v4-flash"
api_key = "sk-..."
base_url = "https://api.deepseek.com/v1"
enable_thinking = true
multimodal = false

[llm.fast]
model = "qwen-flash"
api_key = "sk-..."
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

[llm.vl]
model = "qwen-vl-plus"
api_key = "sk-..."
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

各模型用途：

- `llm.main`：主要推理和对话模型。
- `llm.fast`：轻量任务模型，用于记忆判断、查询改写等步骤。
- `llm.vl`：视觉模型；主模型不支持图片时使用。

如果只使用一个兼容 OpenAI API 的模型服务，可以让多个模型配置使用相同的 `api_key` 和 `base_url`。

## 配置记忆

```toml
[memory]
enabled = true
engine = ""

[memory.embedding]
model = "text-embedding-v3"
api_key = "sk-..."
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

- `memory.enabled`：是否启用记忆功能。
- `memory.engine`：留空时使用默认记忆插件。
- `memory.embedding`：配置向量化服务，接口需要兼容当前项目的 Embedding 调用方式。

## 配置 Telegram

```toml
[channels.telegram]
token = "123456:ABC..."
allow_from = ["your_username"]
```

- `token`：通过 Telegram 的 BotFather 创建机器人后获得。
- `allow_from`：允许与机器人交互的 Telegram 用户名列表。

如果不使用 Telegram，可以在初始化向导中选择其他已经支持的通信渠道，并填写对应配置。

## 启动

```bash
uv run python main.py
```

启动后，通过配置好的通信渠道向机器人发送一条消息，确认模型、通信渠道和记忆服务均能正常工作。
