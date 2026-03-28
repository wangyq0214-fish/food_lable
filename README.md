# 食品标签 OpenClaw Skill（骨架）

本项目支持：
- `classic`：规则占位版工作流
- `graph`：LangGraph 多智能体工作流（百炼）
- `LlamaIndex` 离线建库 + 在线只读检索

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境变量

### OCR（阿里云市场接口）

```bash
setx OCR_AUTH_MODE "APPCODE"
setx OCR_APPCODE "你的appcode"
setx OCR_ENDPOINT "https://gjbsb.market.alicloudapi.com/ocrservice/advanced"
```

### 百炼（Graph 引擎）

```bash
setx DASHSCOPE_API_KEY "你的百炼API Key"
setx DASHSCOPE_MODEL "qwen-plus"
```

> 设置后重开终端。

## 1) 离线构建向量库（先执行一次）

```bash
python -m openclaw_skill.cli --build-index --index-source-dir "data/cleaned" --index-persist-dir "vector_store/food_label_rules"
```

构建成功后会输出：

```text
INDEX: vector_store/food_label_rules
```

## 2) 运行审核

### classic

```bash
python -m openclaw_skill.cli "test/2 标签来找茬/1.1 问题标签.docx" --engine classic
```

### graph（多智能体 + 百炼 + 离线RAG）

```bash
python -m openclaw_skill.cli "test/2 标签来找茬/1.1 问题标签.docx" --engine graph
```

## 调试输出

```bash
python -m openclaw_skill.cli "你的输入.docx" --engine graph --debug-ocr --debug-ocr-file "outputs/my_ocr_raw.json" --debug-parsed-file "outputs/parsed.json"
```

## 支持输入格式

- 文本：`.txt`
- 文档：`.docx`（含内嵌图片 OCR）
- 图片：`.jpg/.jpeg/.png/.bmp/.webp`
