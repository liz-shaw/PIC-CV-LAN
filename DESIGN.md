# PIC-CV-LAN 设计文档

## 1. 产品定位

PIC-CV-LAN 是一个本地图片词汇采集器。核心目标不是做复杂图像理解，而是服务语言学习：把现实物品快速转成英语、中文、德语词条，并进入复习系统。

## 2. 非目标

当前版本暂不追求：

- 复杂场景推理
- 长句图像描述
- 云端多模态大模型
- 自动生成完美德语例句
- 完整桌面客户端

原因：这些会显著增加复杂度，但对 V1 的学习闭环收益不高。

## 3. V1 架构

```text
User Image
   ↓
Gradio UI
   ↓
ObjectDetector / YOLO
   ↓
English labels + confidence
   ↓
VocabStore / vocab.tsv
   ↓
English / 中文 / Deutsch
   ↓
DataFrame display + TSV export
```

## 4. 模块划分

### app.py

负责界面和流程编排。

### src/pic_cv_lan/detector.py

封装 YOLO：

- 加载模型
- 识别图片
- 过滤低置信度结果
- 合并重复标签
- 返回标注图

### src/pic_cv_lan/vocab.py

负责本地词典：

- 加载 `vocab.tsv`
- 查询英文标签
- 未收录词写入 `unknown_words.tsv`

### src/pic_cv_lan/exporter.py

负责导出：

- 创建 `outputs/`
- 写出 `latest_image_vocab.tsv`

## 5. 数据格式

### vocab.tsv

```tsv
English	中文	Deutsch	Note
cat	猫	die Katze	common object
```

### unknown_words.tsv

```tsv
English	中文	Deutsch	Status	Source
remote	待补充	待补充	未处理	image-detect
```

### outputs/latest_image_vocab.tsv

```tsv
English	中文	Deutsch	Confidence	Source
cat	猫	die Katze	0.912	YOLO
```

## 6. 为什么 V1 不使用翻译 API

物体识别的标签本身数量有限，用本地词典更稳定。

尤其是德语，名词需要冠词。机器翻译经常只给 `Katze`，但 Anki 更需要 `die Katze`。

## 7. 后续扩展路线

### V2：生词本管理

- 页面展示 `unknown_words.tsv`
- 允许手动补充中文和德语
- 一键合并回 `vocab.tsv`

### V3：Anki 模板导出

增加导出格式：

```tsv
Front	Back	Tags
cat	猫；die Katze	image_vocab
```

### V4：本地视觉模型兜底

当 YOLO 没识别到结果时，调用本地视觉模型，例如 Ollama 里的视觉模型，生成候选物品描述。

```text
YOLO 成功 → 用 YOLO
YOLO 失败 → 本地视觉模型兜底
```

### V5：桌面应用

可选路线：

- Python + PySide6
- Tauri + Python backend
- Electron + Python backend

不建议 V1 就做桌面端。先验证学习闭环，再做壳。
