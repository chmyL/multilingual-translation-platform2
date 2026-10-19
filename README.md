# 多语言翻译与文本润色平台

一个基于本地大语言模型的多语言文本处理与实验平台，集成翻译、学术润色、口语/书面语转换、翻译质量评估、批量处理和模型/策略对比实验。项目通过 vLLM 提供本地模型推理服务，并使用 FastAPI 提供后端 API，提供 Web 界面进行交互。

## 功能特性

- **多语言翻译**：支持中文、英文、日文、韩文之间配置好的语言方向，包括中英、中日和中韩互译。
- **多种翻译策略**：提供直译、意译和学术风格翻译策略，便于比较不同提示策略的输出。
- **学术文本润色**：支持轻度、中度和重度润色，可针对不同语言调整文本表达。
- **口语/书面语转换**：支持将口语表达改写为更正式的书面语，以及反向转换。
- **质量评估**：集成 BLEU 等自动指标，并支持准确度、流畅度、充分性等维度的人工评分。
- **批量处理**：支持对多段文本进行批量翻译与处理。
- **对比实验**：支持模型、翻译策略、语言方向及领域相关实验，并生成图表和实验报告。
- **领域感知处理**：包含领域分类、术语词典及领域相关策略/消融实验逻辑。
- **结果导出与历史记录**：支持将处理结果导出为 JSON、TXT、Markdown 等格式，并保存历史记录。
- **本地推理**：通过 vLLM 的 OpenAI 兼容 API 调用本地模型，避免将文本默认发送到第三方托管推理服务。

## 技术栈

- **语言**：Python 3.10+
- **后端**：FastAPI、Uvicorn、Pydantic
- **模型推理**：vLLM、Transformers、PyTorch、OpenAI Python SDK（用于访问兼容接口）
- **前端**：HTML、CSS、JavaScript；项目另保留一个可选的 Streamlit 界面
- **自然语言处理与评估**：Sentence Transformers、SacreBLEU、NumPy
- **数据分析与可视化**：Pandas、Matplotlib、Plotly
- **文档解析**：pypdf、python-docx
- **数据存储**：JSON 文件（语料、运行历史、评分与实验结果）

## 项目结构

```text
PythonProject3/
├── server.py                   # FastAPI 服务入口及 REST API
├── config.py                   # 模型、推理服务、语言、路径和实验配置
├── vllm_manager.py              # vLLM 服务启停、模型切换和健康检查
├── translator.py                # 翻译与批量翻译
├── polisher.py                  # 文本润色
├── converter.py                 # 口语/书面语转换
├── evaluator.py                 # 自动评估与人工评分
├── experiment.py                # 模型/策略/语言方向实验、图表与报告
├── domain_aware.py              # 领域感知与消融实验相关逻辑
├── domain_classifier.py         # 领域分类
├── domain_glossary.py           # 领域术语词典
├── sentence_splitter.py         # 长文本切分及特殊 token 处理
├── style_analyzer.py            # 文本风格分析
├── exporter.py                  # 结果导出
├── llm_adapter.py               # 大模型接口适配
├── frontend/
│   ├── app.html                 # Web 前端页面
│   └── app.py                   # 可选的 Streamlit 前端
├── data/
│   ├── corpus/                  # 项目语料与构建后的数据集
│   └── results/                 # 运行生成的历史、评分、实验结果（默认忽略）
├── outputs/                     # 导出的文件（默认忽略）
├── test_examples/               # 功能测试样例与测试说明
├── build_*.py                   # 语料/数据集构建脚本
├── run_real_experiment.py       # 真实语料实验入口
├── DATASETS.md                  # 数据集来源、构建方式与说明
├── requirements.txt             # Python 依赖
└── start.sh                     # WSL/Linux 环境的一键启动脚本（需按环境调整）
```

## 运行环境说明

项目当前配置面向 **Linux/WSL2 + NVIDIA GPU + CUDA 的本地推理环境**。仓库中的模型路径、Conda 路径和项目目录包含特定机器的绝对路径，克隆后需要先修改配置；不能假设在 Windows 原生 Python 环境或其他 Linux 主机上直接运行。

建议环境：

- Python 3.10 或更高版本
- NVIDIA GPU 及兼容的驱动/CUDA 环境
- 已安装并可正常使用的 vLLM
- 项目配置的模型文件已下载到本地

> vLLM 对操作系统、CUDA、PyTorch 和 Python 版本有较强的兼容性要求。建议先按照 vLLM 官方安装说明建立独立环境，再安装项目其他依赖。`requirements.txt` 中的版本下限不等于所有最新版本都能互相兼容。

## 安装与配置

### 1. 获取项目

```bash
git clone <你的 GitHub 仓库地址>
cd PythonProject3
```

### 2. 创建 Python 环境

```bash
conda create -n translation-platform python=3.10 -y
conda activate translation-platform
```

### 3. 安装依赖

先根据本机 CUDA、PyTorch 和系统环境安装兼容版本的 vLLM，再安装项目依赖：

```bash
pip install -r requirements.txt
```

如果只使用 FastAPI 主流程但不需要 Streamlit 旧前端，可以根据实际需要省略 Streamlit；若不运行数据集构建脚本，也可不安装相关可选依赖。

### 4. 修改本地路径

启动前请检查 `config.py` 和 `server.py` 中的路径配置，至少包括：

- `MODELS` 中各模型的本地 `path`
- `DATA_DIR`、`CORPUS_DIR`、`RESULTS_DIR` 和 `OUTPUT_DIR`
- `VLLM_START_CMD_TEMPLATE`、`_VLLM_BIN`、`_VLLM_ENV_BIN` 中的 Conda/vLLM 路径
- `server.py` 中添加到 `sys.path` 的项目目录，以及前端 `frontend/app.html` 的绝对路径
- `start.sh` 中的 Conda 初始化路径、项目目录和系统工具路径

这些路径是当前开发环境配置，不是通用路径。请将它们改为你自己的实际位置后再启动。

### 5. 启动服务

在完成上述配置、确认 vLLM 环境和模型路径正确后，可以直接启动 FastAPI：

```bash
python server.py
```

如果需要显式指定监听地址和端口，也可使用：

```bash
uvicorn server:app --host 127.0.0.1 --port 8501
```

浏览器访问：

```text
http://127.0.0.1:8501
```

项目默认的 vLLM 推理 API 地址为 `http://localhost:8000`。请确认 `config.py` 中的 `VLLM_HOST`、`VLLM_PORT` 与实际服务一致。

`start.sh` 是为特定 WSL/Linux 主机编写的一键启动脚本，其中包含清理进程、Conda 路径和 GPU 进程操作。**检查并修改脚本后再运行**，不要在有其他重要 Python/GPU 任务运行时直接执行。

## 使用流程

1. 启动并确认 vLLM 服务已就绪。
2. 在浏览器打开 Web 页面。
3. 选择所需功能：翻译、润色、口语/书面语转换、质量评估、批量处理或对比实验。
4. 根据界面选择语言方向、翻译策略、润色级别或实验参数。
5. 查看处理结果；需要留存时导出文件或查看历史/实验结果。

具体界面和测试示例可参考 [`test_examples/README.md`](test_examples/README.md)。

## 数据集与实验

项目自带构建语料位于 `data/corpus/`。数据集的字段、来源、下载地址和构建脚本说明见 [`DATASETS.md`](DATASETS.md)。

主要数据集构建脚本包括：

- `build_real_datasets.py`、`build_real_corpus.py`：真实语料相关构建流程
- `build_strategy_corpus.py`：翻译策略实验语料构建
- `build_ablation_from_raw.py`：消融实验语料构建
- `build_conversion_corpus.py`：口语/书面语转换语料构建
- `run_real_experiment.py`：真实语料实验入口

实验会将结果、图表和报告写入 `data/results/`；导出的文件通常写入 `outputs/`。这些属于运行生成物，默认不建议提交到 Git。若要公开实验结论，可以挑选必要的汇总报告和图表，单独放入文档目录或 README。

请在重新分发外部数据集、语料或模型文件前，确认其原始许可证及使用要求。项目代码中的数据集说明不代表所有外部数据均可无限制再分发。

## 评估指标

- **BLEU**：衡量候选译文与参考译文之间的 n-gram 重合情况。
- **METEOR**：项目包含简化的纯 Python 近似实现；请勿将其直接等同于官方完整 METEOR 实现。
- **人工评分**：支持准确度、流畅度和充分性等维度的评分，适合与自动指标结合分析。

自动指标只能从特定角度衡量译文质量，不应单独作为翻译质量的最终结论。

## 注意事项

- 首次加载模型可能需要较长时间，并占用大量显存和磁盘空间。
- 模型权重没有包含在本仓库中，需要自行下载并在 `config.py` 配置路径。
- `data/results/` 和 `outputs/` 包含可能由运行产生的个人数据或实验产物，默认通过 `.gitignore` 排除。
- 当前项目使用 JSON 文件存储部分历史记录和实验数据，不是数据库驱动的多用户服务。
- 使用本地模型并不自动意味着所有数据都绝对安全；请仍然妥善管理日志、导出文件和主机访问权限。

## 后续可改进方向

- 将机器相关路径迁移到环境变量或独立的本地配置文件。
- 为 API 增加统一异常处理、日志配置和输入校验。
- 将实验配置、数据集版本和评估结果结构化，方便复现实验。
- 增加自动化测试与部署说明。
- 为外部语料、模型和实验结果补充更完整的许可证与复现信息。

## 许可证

当前项目尚未在仓库中明确指定开源许可证。公开发布前，请根据代码、数据集和模型的授权情况添加合适的 `LICENSE` 文件。
