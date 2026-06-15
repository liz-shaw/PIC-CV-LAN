# PIC-CV-LAN

一个本地运行的图片物品识别与三语词汇生成工具：上传/拍照图片，识别主要物品，并输出 English / 中文 / Deutsch，可导出 TSV 供 Anki 使用。

## 目标

```text
图片输入
→ YOLO 常见物体检测
→ 检测框后处理去重
→ Ollama 本地视觉模型补充广义物品词条
→ 英中德词条整理
→ 显示结果
→ 导出 TSV / 记录未收录词
```

## 当前版本：V1

- 不需要 OpenAI / Claude / Gemini 等 AI token
- 支持双引擎：YOLO 快速检测 + Ollama 本地视觉模型广义识别
- YOLO 有检测框，但类别少
- Ollama 识别范围更广，但没有精准检测框，且需要本地安装视觉模型
- 默认使用 CPU，适合没有 NVIDIA GPU 的服务器
- 支持检测框后处理去重，减少同一物品重复识别
- 页面默认是“一键使用”，高级参数默认收起
- 支持“混合模式 / YOLO 快速模式 / Ollama 广义识别”
- 支持“日常模式 / 更多物品 / 减少重复 / 弱 CPU 快速 / 自定义”
- 使用本地 `vocab.tsv` 做英中德映射，Ollama 也可直接补充英中德词条
- 支持图片上传与摄像头输入
- 支持导出 Anki 可用的 TSV
- 未收录词会写入 `unknown_words.tsv`

## 安装

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 运行

基础运行：

```bash
python app.py
```

显式指定 CPU：

```bash
python app.py --device cpu
```

服务器给局域网访问：

```bash
python app.py --device cpu --imgsz 640 --host 0.0.0.0 --port 7860
```

首次运行会自动下载 YOLO 模型文件，例如 `yolo11n.pt`。之后 YOLO 部分可以本地运行，不需要 AI token。

## 推荐使用流程

```text
1. 上传图片
2. 识别引擎选择：混合模式：YOLO + Ollama，推荐
3. YOLO 模式保持：日常模式
4. 点击：识别并生成词条
5. 下载 TSV 导入 Anki
```

如果没有安装 Ollama，混合模式会提示错误，但 YOLO 仍然可用。此时可以先选择：

```text
常见物体检测：快，有框
```

## 为什么要加 Ollama？

纯 YOLO 的问题是类别上限太低。普通 YOLO 模型擅长识别 COCO 常见物体，例如 person、cat、dog、cup、chair、book、keyboard、cell phone、car 等，但它不是开放词汇视觉模型。

Ollama 本地视觉模型更适合做：

```text
现实图片 → 尽可能列出主要可见物品 → 生成英语/中文/德语词条
```

代价是：

```text
速度更慢
需要本地安装模型
没有精准检测框
少量结果需要人工复核
```

## 启用 Ollama 广义识别

安装并启动 Ollama 后，拉取一个视觉模型。默认配置使用：

```bash
ollama pull gemma3:4b
```

启动 Ollama 服务：

```bash
ollama serve
```

运行项目：

```bash
python app.py --device cpu --ollama-model gemma3:4b
```

如果你使用其他视觉模型，可以改：

```bash
python app.py --ollama-model 你的模型名
```

Ollama 的本地服务默认地址是：

```text
http://localhost:11434
```

如果你的 Ollama 在别的机器上：

```bash
python app.py --ollama-host http://服务器IP:11434
```

注意：不要把 Ollama 端口裸露到公网。

## 三种识别引擎

```text
混合模式：YOLO + Ollama，推荐
    适合日常使用。YOLO 给框，Ollama 补充更多物品词条。

常见物体检测：快，有框
    适合没有 Ollama、只想快、只想看框。

广义识别：范围大，无框，需要 Ollama
    适合只想收集词汇，不关心检测框。
```

## 重复识别怎么处理？

当前版本做了两层处理：

1. **检测框去重**：对高度重叠的检测框做 IoU 后处理，减少同一物品被框多次。
2. **词条合并**：学习模式下，同类物品合并成一行，并用 `Count` 表示数量。

如果同一个物品还是重复出现：

```text
先选“减少重复”
还不行：选“自定义”
去重方式：强力去重
IoU：0.55～0.65
```

## 为什么识别的物品少？

常见原因：

1. **YOLO 是固定类别检测器**：类别范围有限。
2. **没有启用 Ollama 广义识别**：只用 YOLO 时，很多生活物品不会出现。
3. **同类物品默认合并**：图里有 5 本书，学习模式下只显示一行 `book`，但 `Count` 会显示数量。
4. **图片尺寸过小**：`imgsz=320` 更快，但小物体容易漏；`imgsz=640/960` 更可能识别小物体，但 CPU 更慢。

如果你想识别插线板、眼药水、护肤品、具体品牌商品、桌面小物件，优先启用混合模式或广义识别。

## CPU 服务器说明

没有 NVIDIA GPU 也可以运行。CPU 模式更慢，但对“上传一张图 → 识别几个主要物品 → 生成词条”这个学习场景是够用的。

如果服务器性能较弱：

```text
YOLO 部分选择：弱 CPU 快速
Ollama 部分选择较小视觉模型
广义识别最多返回物品数调小，比如 10～15
```

不要一开始处理视频流或高并发请求。这个项目 V1 的定位是个人学习工具，不是生产级视觉服务。

## 词典格式

`vocab.tsv` 使用制表符分隔：

```tsv
English	中文	Deutsch	Note
cat	猫	die Katze	common object
cup	杯子	die Tasse	common object
```

德语名词建议带冠词：`der / die / das`。

## Anki 导出字段

当前导出字段：

```tsv
English	中文	Deutsch	Count	Confidence	Source
```

可以直接导入 Anki，也可以后续扩展为：

```tsv
Front	Back	Tags
cat	猫；die Katze	image_vocab
```

## 推荐路线

```text
V1：YOLO + Ollama 混合识别 + 三语词典 + TSV 导出
V2：生词本 unknown_words.tsv 管理
V3：Anki 专用模板导出
V4：开放词汇检测模型 YOLO-World / GroundingDINO，用文本提示生成检测框
```
