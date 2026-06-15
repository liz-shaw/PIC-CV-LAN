from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

from src.pic_cv_lan.detector import ObjectDetector
from src.pic_cv_lan.exporter import export_rows_to_tsv
from src.pic_cv_lan.vocab import VocabStore, append_unknown_word


VOCAB_PATH = Path("vocab.tsv")
UNKNOWN_PATH = Path("unknown_words.tsv")
OUTPUT_PATH = Path("outputs/latest_image_vocab.tsv")

DISPLAY_COLUMNS = ["English", "中文", "Deutsch", "Count", "Confidence"]
EXPORT_COLUMNS = ["English", "中文", "Deutsch", "Count", "Confidence", "Source"]
IMAGE_SIZE_CHOICES = [320, 640, 960, 1280]
CUSTOM_PRESET = "自定义"

PRESETS = {
    "日常模式": {
        "conf": 0.25,
        "imgsz": 640,
        "duplicate_mode": "same_label",
        "iou": 0.70,
        "max_det": 100,
        "note": "适合普通拍照，速度和稳定性比较均衡。",
    },
    "更多物品": {
        "conf": 0.15,
        "imgsz": 960,
        "duplicate_mode": "same_label",
        "iou": 0.60,
        "max_det": 180,
        "note": "适合杂物多、小物体多的图片，CPU 会慢一些。",
    },
    "减少重复": {
        "conf": 0.35,
        "imgsz": 640,
        "duplicate_mode": "any_label",
        "iou": 0.65,
        "max_det": 80,
        "note": "适合重复框、误识别比较多的图片。",
    },
    "弱 CPU 快速": {
        "conf": 0.25,
        "imgsz": 320,
        "duplicate_mode": "same_label",
        "iou": 0.70,
        "max_det": 60,
        "note": "适合没有 GPU 的弱服务器，速度优先。",
    },
}

DUPLICATE_MODE_MAP = {
    "保留全部": "none",
    "标准去重": "same_label",
    "强力去重": "any_label",
}

APP_CSS = """
:root {
    --pic-bg: #f7f8fb;
    --pic-panel: rgba(255, 255, 255, 0.94);
    --pic-border: rgba(15, 23, 42, 0.10);
    --pic-text: #0f172a;
    --pic-muted: #64748b;
    --pic-primary: #2563eb;
    --pic-good: #16a34a;
    --pic-warn: #d97706;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background: radial-gradient(circle at top left, #dbeafe 0, transparent 30%),
                radial-gradient(circle at top right, #fef3c7 0, transparent 28%),
                var(--pic-bg) !important;
    color: var(--pic-text) !important;
}

.pic-hero {
    border: 1px solid var(--pic-border);
    border-radius: 26px;
    padding: 24px 28px;
    margin: 8px 0 16px 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.97), rgba(239,246,255,0.90));
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}

.pic-hero-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    flex-wrap: wrap;
}

.pic-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.pic-logo {
    width: 52px;
    height: 52px;
    border-radius: 18px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-size: 27px;
    box-shadow: 0 12px 26px rgba(37, 99, 235, 0.24);
}

.pic-title {
    font-size: 32px;
    line-height: 1.05;
    font-weight: 880;
    letter-spacing: -0.04em;
    margin: 0;
}

.pic-subtitle {
    margin: 7px 0 0 0;
    color: var(--pic-muted);
    font-size: 15px;
}

.pic-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.pic-badge {
    border: 1px solid rgba(37, 99, 235, 0.16);
    background: rgba(219, 234, 254, 0.78);
    color: #1d4ed8;
    padding: 8px 11px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 750;
}

.pic-card {
    border: 1px solid var(--pic-border) !important;
    border-radius: 24px !important;
    padding: 16px !important;
    background: var(--pic-panel) !important;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06) !important;
}

.pic-section-title {
    font-size: 18px;
    font-weight: 860;
    margin: 0 0 7px 0;
    letter-spacing: -0.02em;
}

.pic-section-note {
    color: var(--pic-muted);
    font-size: 13px;
    margin: 0 0 12px 0;
    line-height: 1.55;
}

.pic-primary-button button {
    border-radius: 16px !important;
    padding: 14px 18px !important;
    font-size: 16px !important;
    font-weight: 850 !important;
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    border: none !important;
    box-shadow: 0 14px 26px rgba(37, 99, 235, 0.20) !important;
}

.pic-summary {
    border: 1px solid var(--pic-border);
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.90);
    padding: 16px;
    margin: 0;
}

.pic-summary-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-top: 12px;
}

.pic-metric {
    border-radius: 16px;
    padding: 12px;
    background: #f8fafc;
    border: 1px solid rgba(15, 23, 42, 0.08);
}

.pic-metric-value {
    font-size: 22px;
    line-height: 1.1;
    font-weight: 900;
    letter-spacing: -0.03em;
}

.pic-metric-label {
    color: var(--pic-muted);
    font-size: 12px;
    margin-top: 3px;
}

.pic-success {
    color: var(--pic-good);
    font-weight: 850;
}

.pic-warning {
    color: var(--pic-warn);
    font-weight: 850;
}

.pic-empty {
    color: var(--pic-muted);
    font-size: 14px;
    line-height: 1.7;
}

.pic-mode-help {
    border: 1px solid rgba(37, 99, 235, 0.12);
    background: rgba(239, 246, 255, 0.72);
    color: #1e3a8a;
    border-radius: 16px;
    padding: 12px 14px;
    font-size: 13px;
    line-height: 1.55;
    margin: 10px 0 12px 0;
}

.pic-footer {
    color: var(--pic-muted);
    text-align: center;
    font-size: 12px;
    margin-top: 18px;
    padding-bottom: 18px;
}

@media (max-width: 820px) {
    .pic-title { font-size: 26px; }
    .pic-summary-grid { grid-template-columns: 1fr; }
}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PIC-CV-LAN local image vocabulary app")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda:0, 0, etc. Default: cpu")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model file/name. Default: yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Default confidence threshold. Default: 0.25")
    parser.add_argument("--imgsz", type=int, default=640, help="Default inference image size. Default: 640")
    parser.add_argument("--host", default=None, help="Server host. Use 0.0.0.0 for LAN/server access")
    parser.add_argument("--port", type=int, default=None, help="Server port, for example 7860")
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    return parser


args = build_parser().parse_args()

detector = ObjectDetector(
    model_name=args.model,
    confidence_threshold=args.conf,
    device=args.device,
    image_size=args.imgsz,
)
vocab_store = VocabStore(VOCAB_PATH)


def build_status_html(
    *,
    row_count: int,
    instance_count: int,
    unique_word_count: int,
    unknown_count: int,
    output_file: Path | None,
    preset: str = "日常模式",
    confidence_threshold: float | None = None,
    image_size: int | None = None,
    duplicate_mode_label: str = "标准去重",
    result_mode: str = "学习模式：合并同类",
    has_image: bool = True,
) -> str:
    if not has_image:
        return """
        <div class="pic-summary">
            <div class="pic-empty">上传图片后点一次“识别并生成词条”即可。默认已经开启去重和学习模式。</div>
        </div>
        """

    conf_text = "默认" if confidence_threshold is None else f"{confidence_threshold:.2f}"
    imgsz_text = "默认" if image_size is None else str(image_size)

    if row_count == 0:
        return f"""
        <div class="pic-summary">
            <div class="pic-warning">没有识别到足够可靠的常见物品。</div>
            <div class="pic-empty">下一步建议：先切到“更多物品”；还不行再换更清晰、主体更近的图片。当前：{preset}，{duplicate_mode_label}，conf={conf_text}，imgsz={imgsz_text}。</div>
            <div class="pic-summary-grid">
                <div class="pic-metric"><div class="pic-metric-value">0</div><div class="pic-metric-label">词条</div></div>
                <div class="pic-metric"><div class="pic-metric-value">0</div><div class="pic-metric-label">实例</div></div>
                <div class="pic-metric"><div class="pic-metric-value">{conf_text}</div><div class="pic-metric-label">置信度</div></div>
                <div class="pic-metric"><div class="pic-metric-value">{imgsz_text}</div><div class="pic-metric-label">图片尺寸</div></div>
            </div>
        </div>
        """

    export_text = str(output_file) if output_file else "未生成"
    unknown_text = "全部已收录" if unknown_count == 0 else f"{unknown_count} 个待补充"
    unknown_class = "pic-success" if unknown_count == 0 else "pic-warning"

    return f"""
    <div class="pic-summary">
        <div class="pic-success">识别完成。重复框已做后处理去重，结果可直接下载 TSV。</div>
        <div class="pic-summary-grid">
            <div class="pic-metric"><div class="pic-metric-value">{row_count}</div><div class="pic-metric-label">词条行</div></div>
            <div class="pic-metric"><div class="pic-metric-value">{instance_count}</div><div class="pic-metric-label">检测实例</div></div>
            <div class="pic-metric"><div class="pic-metric-value">{unique_word_count}</div><div class="pic-metric-label">不同英文词</div></div>
            <div class="pic-metric"><div class="pic-metric-value {unknown_class}">{unknown_text}</div><div class="pic-metric-label">词典覆盖</div></div>
        </div>
        <div class="pic-empty" style="margin-top: 12px;">模式：{preset}；输出：{result_mode}；去重：{duplicate_mode_label}；设备：{args.device}；conf={conf_text}；imgsz={imgsz_text}；导出：{export_text}</div>
    </div>
    """


def resolve_config(
    preset: str,
    custom_conf: float,
    custom_imgsz: int,
    custom_duplicate_mode_label: str,
    custom_iou: float,
    custom_max_det: int,
) -> dict[str, object]:
    if preset != CUSTOM_PRESET:
        config = PRESETS[preset].copy()
        config["duplicate_mode_label"] = {
            "none": "保留全部",
            "same_label": "标准去重",
            "any_label": "强力去重",
        }[str(config["duplicate_mode"])]
        return config

    duplicate_mode = DUPLICATE_MODE_MAP[custom_duplicate_mode_label]
    return {
        "conf": float(custom_conf),
        "imgsz": int(custom_imgsz),
        "duplicate_mode": duplicate_mode,
        "duplicate_mode_label": custom_duplicate_mode_label,
        "iou": float(custom_iou),
        "max_det": int(custom_max_det),
        "note": "使用高级参数。",
    }


def detect_and_translate(
    image: Image.Image | None,
    preset: str,
    result_mode: str,
    custom_conf: float,
    custom_imgsz: int,
    custom_duplicate_mode_label: str,
    custom_iou: float,
    custom_max_det: int,
):
    if image is None:
        empty_df = pd.DataFrame(columns=DISPLAY_COLUMNS)
        return None, build_status_html(row_count=0, instance_count=0, unique_word_count=0, unknown_count=0, output_file=None, has_image=False), empty_df, None

    config = resolve_config(
        preset=preset,
        custom_conf=custom_conf,
        custom_imgsz=custom_imgsz,
        custom_duplicate_mode_label=custom_duplicate_mode_label,
        custom_iou=custom_iou,
        custom_max_det=custom_max_det,
    )

    unique_labels = result_mode == "学习模式：合并同类"
    detections, annotated_image = detector.detect(
        image,
        confidence_threshold=float(config["conf"]),
        image_size=int(config["imgsz"]),
        unique_labels=unique_labels,
        max_detections=int(config["max_det"]),
        duplicate_mode=str(config["duplicate_mode"]),
        iou_threshold=float(config["iou"]),
    )

    rows: list[dict[str, object]] = []
    unknown_words: set[str] = set()

    for detection in detections:
        entry = vocab_store.lookup(detection.english)
        if entry is None:
            append_unknown_word(detection.english, UNKNOWN_PATH)
            chinese = "待补充"
            german = "待补充"
            unknown_words.add(detection.english)
        else:
            chinese = entry.chinese
            german = entry.german

        rows.append(
            {
                "English": detection.english,
                "中文": chinese,
                "Deutsch": german,
                "Count": detection.count,
                "Confidence": round(detection.confidence, 3),
                "Source": f"YOLO/{args.device}",
            }
        )

    output_file = export_rows_to_tsv(rows, OUTPUT_PATH, fieldnames=EXPORT_COLUMNS) if rows else None
    display_df = pd.DataFrame([{key: row.get(key, "") for key in DISPLAY_COLUMNS} for row in rows], columns=DISPLAY_COLUMNS)
    instance_count = int(sum(int(row.get("Count", 1)) for row in rows))
    unique_word_count = len({str(row.get("English", "")) for row in rows if row.get("English")})

    status_html = build_status_html(
        row_count=len(rows),
        instance_count=instance_count,
        unique_word_count=unique_word_count,
        unknown_count=len(unknown_words),
        output_file=output_file,
        preset=preset,
        confidence_threshold=float(config["conf"]),
        image_size=int(config["imgsz"]),
        duplicate_mode_label=str(config["duplicate_mode_label"]),
        result_mode=result_mode,
        has_image=True,
    )
    return annotated_image, status_html, display_df, str(output_file) if output_file else None


hero_html = """
<div class="pic-hero">
    <div class="pic-hero-top">
        <div class="pic-brand">
            <div class="pic-logo">📷</div>
            <div>
                <h1 class="pic-title">PIC-CV-LAN</h1>
                <p class="pic-subtitle">上传图片 → 识别物品 → 生成 English / 中文 / Deutsch → 导出 Anki TSV</p>
            </div>
        </div>
        <div class="pic-badges">
            <span class="pic-badge">No Token</span>
            <span class="pic-badge">Local YOLO</span>
            <span class="pic-badge">Duplicate Clean</span>
            <span class="pic-badge">Anki Ready</span>
        </div>
    </div>
</div>
"""

with gr.Blocks(title="PIC-CV-LAN", css=APP_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(hero_html)

    with gr.Row(equal_height=True):
        with gr.Column(scale=5, elem_classes=["pic-card"]):
            gr.HTML('<div class="pic-section-title">1. 上传图片</div><div class="pic-section-note">默认不用调参数。照片里主体越近、背景越干净，结果越稳。</div>')
            image_input = gr.Image(
                type="pil",
                sources=["upload", "webcam"],
                label="上传图片或使用摄像头",
                height=360,
            )

            gr.HTML('<div class="pic-section-title">2. 选择使用模式</div>')
            preset_radio = gr.Radio(
                choices=list(PRESETS.keys()) + [CUSTOM_PRESET],
                value="日常模式",
                label="识别模式",
                info="大多数情况选日常模式；重复多选减少重复；想多识别选更多物品。",
            )
            result_mode_radio = gr.Radio(
                choices=["学习模式：合并同类", "检查模式：显示每个实例"],
                value="学习模式：合并同类",
                label="输出方式",
                info="背单词建议合并同类；排查识别问题再看每个实例。",
            )

            gr.HTML('<div class="pic-mode-help">推荐流程：先用“日常模式”。如果同一个东西被框了很多次，切“减少重复”。如果识别太少，切“更多物品”。</div>')
            run_button = gr.Button("3. 识别并生成词条", variant="primary", elem_classes=["pic-primary-button"])

            with gr.Accordion("高级参数：一般不用打开", open=False):
                custom_conf = gr.Slider(
                    minimum=0.05,
                    maximum=0.80,
                    step=0.05,
                    value=args.conf,
                    label="自定义置信度",
                    info="只在选择“自定义”时生效。越低结果越多，误识别也越多。",
                )
                custom_imgsz = gr.Radio(
                    choices=IMAGE_SIZE_CHOICES,
                    value=args.imgsz if args.imgsz in IMAGE_SIZE_CHOICES else 640,
                    label="自定义图片尺寸",
                    info="只在选择“自定义”时生效。越大越慢，但小物体更容易出现。",
                )
                custom_duplicate_mode = gr.Radio(
                    choices=list(DUPLICATE_MODE_MAP.keys()),
                    value="标准去重",
                    label="自定义去重方式",
                    info="只在选择“自定义”时生效。强力去重会压掉重叠很高的不同标签。",
                )
                custom_iou = gr.Slider(
                    minimum=0.30,
                    maximum=0.95,
                    step=0.05,
                    value=0.70,
                    label="自定义去重阈值 IoU",
                    info="只在选择“自定义”时生效。越低去重越狠。",
                )
                custom_max_det = gr.Slider(
                    minimum=5,
                    maximum=300,
                    step=5,
                    value=100,
                    label="自定义最大检测数量",
                    info="只在选择“自定义”时生效。密集场景可调大。",
                )

        with gr.Column(scale=5, elem_classes=["pic-card"]):
            gr.HTML('<div class="pic-section-title">检测预览</div><div class="pic-section-note">这里显示已经后处理去重的框。框太多：选“减少重复”；框太少：选“更多物品”。</div>')
            annotated_output = gr.Image(label="检测标注图", height=560)

    with gr.Row(equal_height=True):
        with gr.Column(scale=4, elem_classes=["pic-card"]):
            gr.HTML('<div class="pic-section-title">运行状态</div>')
            status_output = gr.HTML(
                build_status_html(row_count=0, instance_count=0, unique_word_count=0, unknown_count=0, output_file=None, has_image=False)
            )
            file_output = gr.File(label="下载 Anki TSV")

        with gr.Column(scale=6, elem_classes=["pic-card"]):
            gr.HTML('<div class="pic-section-title">词条结果</div><div class="pic-section-note">Count 表示同类数量。表格故意隐藏 Source，减少噪音；TSV 里仍会保留来源。</div>')
            table_output = gr.Dataframe(
                label="English / 中文 / Deutsch",
                headers=DISPLAY_COLUMNS,
                interactive=False,
                row_count=(8, "dynamic"),
                col_count=(5, "fixed"),
            )

    with gr.Accordion("使用建议", open=False):
        gr.Markdown(
            f"""
            - **重复识别多**：选“减少重复”。如果还多，选择“自定义”，去重方式选“强力去重”，IoU 调到 `0.55～0.65`。
            - **识别物品少**：选“更多物品”。如果 CPU 太慢，再回到“日常模式”。
            - **背单词导出**：输出方式选“学习模式：合并同类”。
            - **排查模型到底识别了几个框**：输出方式选“检查模式：显示每个实例”。
            - 当前启动：模型 `{args.model}`；设备 `{args.device}`；默认 conf `{args.conf}`；默认 imgsz `{args.imgsz}`。
            """
        )

    gr.HTML('<div class="pic-footer">PIC-CV-LAN · Local-first vocabulary capture tool · Less knobs, cleaner results.</div>')

    run_button.click(
        fn=detect_and_translate,
        inputs=[
            image_input,
            preset_radio,
            result_mode_radio,
            custom_conf,
            custom_imgsz,
            custom_duplicate_mode,
            custom_iou,
            custom_max_det,
        ],
        outputs=[annotated_output, status_output, table_output, file_output],
    )


if __name__ == "__main__":
    demo.queue().launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )
