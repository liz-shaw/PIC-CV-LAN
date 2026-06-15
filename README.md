# PIC-CV-LAN

一个本地运行的图片物品识别与三语词汇生成工具：上传/拍照图片，识别主要物品，并输出 English / 中文 / Deutsch，可导出 TSV 供 Anki 使用。

## 目标

```text
图片输入
→ 本地 YOLO 物体检测
→ 得到英文标签
→ 查询本地英中德词典
→ 显示识别结果
→ 导出 TSV / 记录未收录词
```

## 当前版本：V1

- 不需要 OpenAI / Claude / Gemini 等 AI token
- 使用本地 YOLO 模型做物体检测
- 默认使用 CPU，适合没有 NVIDIA GPU 的服务器
- 使用本地 `vocab.tsv` 做英中德映射
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

CPU 默认模式：

```bash
python app.py
```

显式指定 CPU：

```bash
python app.py --device cpu
```

如果以后机器有 NVIDIA GPU / CUDA，可以指定：

```bash
python app.py --device cuda:0
```

降低输入尺寸可以提升 CPU 推理速度：

```bash
python app.py --device cpu --imgsz 320
```

然后打开终端中显示的本地网址。

首次运行会自动下载 YOLO 模型文件，例如 `yolo11n.pt`。之后可以本地运行，不需要 AI token。

## CPU 服务器说明

没有 NVIDIA GPU 也可以运行。CPU 模式更慢，但对“上传一张图 → 识别几个主要物品 → 生成词条”这个学习场景是够用的。

如果服务器性能较弱，优先使用：

```bash
python app.py --device cpu --imgsz 320 --conf 0.35
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
English	中文	Deutsch	Confidence	Source
```

可以直接导入 Anki，也可以后续扩展为：

```tsv
Front	Back	Tags
cat	猫；die Katze	image_vocab
```

## 推荐路线

```text
V1：图片识别 + 三语词典 + TSV 导出
V2：生词本 unknown_words.tsv 管理
V3：Anki 专用模板导出
V4：接入本地视觉大模型作为 YOLO 识别失败时的兜底
```
