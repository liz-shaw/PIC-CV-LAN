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

APP_CSS = """
:root {
    --pic-bg: #f7f8fb;
    --pic-panel: rgba(255, 255, 255, 0.92);
    --pic-border: rgba(15, 23, 42, 0.10);
    --pic-text: #0f172a;
    --pic-muted: #64748b;
    --pic-primary: #2563eb;
    --pic-primary-soft: #dbeafe;
    --pic-good: #16a34a;
    --pic-warn: #d97706;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background: radial-gradient(circle at top left, #dbeafe 0, transparent 32%),
                radial-gradient(circle at top right, #fef3c7 0, transparent 28%),
                var(--pic-bg) !important;
    color: var(--pic-text) !important;
}

.pic-hero {
    border: 1px solid var(--pic-border);
    border-radius: 28px;
    padding: 28px 30px;
    margin: 8px 0 18px 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,246,255,0.90));
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
    width: 54px;
    height: 54px;
    border-radius: 18px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-size: 28px;
    box-shadow: 0 12px 26px rgba(37, 99, 235, 0.26);
}

.pic-title {
    font-size: 34px;
    line-height: 1.05;
    font-weight: 860;
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
    font-weight: 700;
}

.pic-steps {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-top: 22px;
}

.pic-step {
    border: 1px solid var(--pic-border);
    background: rgba(255,255,255,0.80);
    border-radius: 18px;
    padding: 13px 14px;
}

.pic-step-num {
    font-size: 12px;
    color: var(--pic-primary);
    font-weight: 800;
    margin-bottom: 5px;
}

.pic-step-title {
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 2px;
}

.pic-step-desc {
    color: var(--pic-muted);
    font-size: 12px;
    line-height: 1.45;
}

.pic-card {
    border: 1px solid var(--pic-border) !important;
    border-radius: 24px !important;
    padding: 16px !important;
    background: var(--pic-panel) !important;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06) !important;
}

.pic-section-title {
    font-size: 17px;
    font-weight: 850;
    margin: 0 0 10px 0;
    letter-spacing: -0.02em;
}

.pic-section-note {
    color: var(--pic-muted);
    font-size: 13px;
    margin: -2px 0 12px 0;
}

.pic-primary-button button {
    border-radius: 16px !important;
    padding: 13px 18px !important;
    font-size: 16px !important;
    font-weight: 850 !important;
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    border: none !important;
    box-shadow: 0 14px 26px rgba(37, 99, 235, 0.20) !important;
}

.pic-summary {
    border: 1px solid var(--pic-border);
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.88);
    padding: 16px;
    margin: 0;
}

.pic-summary-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
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
    font-size: 24px;
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

.pic-footer {
    color: var(--pic-muted);
    text-align: center;
    font-size: 12px;
    margin-top: 18px;
    padding-bottom: 18px;
}

@media (max-width: 820px) {
    .pic-title {
        font-size: 27px;
    }
    .pic-steps {
        grid-template-columns: 1fr;
    }
    .pic-summary-grid {
        grid-template-columns: 1fr;
    }
}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PIC-CV-LAN local image vocabulary app")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda:0, 0, etc. Default: cpu")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model file/name. Default: yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold. Default: 0.35")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size. Default: 640")
    parser.add_argument("--host", default=None, help="Server host. Use 0.0.0.0 for LAN/server access")
    parser.add_argument("--port", type=int, default=None, help="Server port, for example 7860")
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    return parser


args = build_parser().parse_args()

# Load once when the app starts. Default device is CPU for non-NVIDIA servers.
detector = ObjectDetector(
    model_name=args.model,
    confidence_threshold=args.conf,
    device=args.device,
    image_size=args.imgsz,
)
vocab_store = VocabStore(VOCAB_PATH)


EMPTY_COLUMNS = ["English", "中文", "Deutsch", "Confidence", "Source"]


def build_status_html(
    *,
    total: int,
    unknown_count: int,
    output_file: Path | None,
    has_image: bool = True,
) -> str:
    if not has_image:
        return """
        <div class="pic-summary">
            <div class="pic-empty">请先上传图片或打开摄像头拍照。建议拍清楚主体，背景越干净越好。</div>
        </div>
        """

    if total == 0:
        return f"""
        <div class="pic-summary">
            <div class="pic-warning">没有识别到置信度足够高的常见物品。</div>
            <div class="pic-empty">可以换一张更清晰的图片，或者降低阈值。当前设备：{args.device}，输入尺寸：{args.imgsz}。</div>
            <div class="pic-summary-grid">
                <div class="pic-metric"><div class="pic-metric-value">0</div><div class="pic-metric-label">识别物品</div></div>
                <div class="pic-metric"><div class="pic-metric-value">{unknown_count}</div><div class="pic-metric-label">未收录词</div></div>
                <div class="pic-metric"><div class="pic-metric-value">CPU</div><div class="pic-metric-label">默认友好模式</div></div>
            </div>
        </div>
        """

    export_text = str(output_file) if output_file else "未生成"
    unknown_text = "全部已收录" if unknown_count == 0 else f"{unknown_count} 个待补充"
    unknown_class = "pic-success" if unknown_count == 0 else "pic-warning"

    return f"""
    <div class="pic-summary">
        <div class="pic-success">识别完成。结果已经生成，可以下载 TSV 导入 Anki。</div>
        <div class="pic-summary-grid">
            <div class="pic-metric"><div class="pic-metric-value">{total}</div><div class="pic-metric-label">识别物品</div></div>
            <div class="pic-metric"><div class="pic-metric-value {unknown_class}">{unknown_text}</div><div class="pic-metric-label">词典覆盖</div></div>
            <div class="pic-metric"><div class="pic-metric-value">{args.device}</div><div class="pic-metric-label">推理设备</div></div>
        </div>
        <div class="pic-empty" style="margin-top: 12px;">导出文件：{export_text}</div>
    </div>
    """


def detect_and_translate(image: Image.Image | None):
    if image is None:
        empty_df = pd.DataFrame(columns=EMPTY_COLUMNS)
        return None, build_status_html(total=0, unknown_count=0, output_file=None, has_image=False), empty_df, None

    detections, annotated_image = detector.detect(image)

    rows: list[dict[str, object]] = []
    unknown_count = 0

    for detection in detections:
        entry = vocab_store.lookup(detection.english)
        if entry is None:
            append_unknown_word(detection.english, UNKNOWN_PATH)
            chinese = "待补充"
            german = "待补充"
            unknown_count += 1
        else:
            chinese = entry.chinese
            german = entry.german

        rows.append(
            {
                "English": detection.english,
                "中文": chinese,
                "Deutsch": german,
                "Confidence": round(detection.confidence, 3),
                "Source": f"YOLO/{args.device}",
            }
        )

    if rows:
        output_file = export_rows_to_tsv(rows, OUTPUT_PATH)
    else:
        output_file = None

    df = pd.DataFrame(rows, columns=EMPTY_COLUMNS)
    status_html = build_status_html(
        total=len(rows),
        unknown_count=unknown_count,
        output_file=output_file,
        has_image=True,
    )
    return annotated_image, status_html, df, str(output_file) if output_file else None


hero_html = f"""
<div class="pic-hero">
    <div class="pic-hero-top">
        <div class="pic-brand">
            <div class="pic-logo">📷</div>
            <div>
                <h1 class="pic-title">PIC-CV-LAN</h1>
                <p class="pic-subtitle">本地图片物品识别 · 英中德词汇生成 · Anki TSV 导出</p>
            </div>
        </div>
        <div class="pic-badges">
            <span class="pic-badge">No AI Token</span>
            <span class="pic-badge">CPU First</span>
            <span class="pic-badge">YOLO Local</span>
            <span class="pic-badge">Anki Ready</span>
        </div>
    </div>
    <div class="pic-steps">
        <div class="pic-step"><div class="pic-step-num">STEP 01</div><div class="pic-step-title">上传图片</div><div class="pic-step-desc">选择照片，或用摄像头拍摄现实物品。</div></div>
        <div class="pic-step"><div class="pic-step-num">STEP 02</div><div class="pic-step-title">本地识别</div><div class="pic-step-desc">YOLO 在本机/服务器上识别主要物体。</div></div>
        <div class="pic-step"><div class="pic-step-num">STEP 03</div><div class="pic-step-title">三语映射</div><div class="pic-step-desc">查询 vocab.tsv，输出 English / 中文 / Deutsch。</div></div>
        <div class="pic-step"><div class="pic-step-num">STEP 04</div><div class="pic-step-title">导出复习</div><div class="pic-step-desc">下载 TSV，导入 Anki 做图像词汇积累。</div></div>
    </div>
</div>
"""

with gr.Blocks(title="PIC-CV-LAN", css=APP_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(hero_html)

    with gr.Row(equal_height=True):
        with gr.Column(scale=5, elem_classes=["pic-card"]):
            gr.HTML('<div class="pic-section-title">输入图片</div><div class="pic-section-note">主体越清楚，识别越稳。CPU 服务器建议先用单张图片，不要直接上视频流。</div>')
            image_input = gr.Image(
                type="pil",
                sources=["upload", "webcam"],
                label="上传图片或使用摄像头",
                height=430,
            )
            run_button = gr.Button("开始识别并生成词条", variant="primary", elem_classes=["pic-primary-button"])

        with gr.Column(scale=5, elem_classes=["pic-card"]):
            gr.HTML('<div class="pic-section-title">检测预览</div><div class="pic-section-note">识别到的物体会在图片上标注。没有框不代表失败，可能是置信度不足或类别不在模型内。</div>')
            annotated_output = gr.Image(label="检测标注图", height=430)

    with gr.Row(equal_height=True):
        with gr.Column(scale=4, elem_classes=["pic-card"]):
            gr.HTML('<div class="pic-section-title">运行状态</div>')
            status_output = gr.HTML(
                build_status_html(total=0, unknown_count=0, output_file=None, has_image=False)
            )
            file_output = gr.File(label="下载 Anki TSV")

        with gr.Column(scale=6, elem_classes=["pic-card"]):
            gr.HTML('<div class="pic-section-title">三语词条结果</div><div class="pic-section-note">德语名词默认带冠词，适合直接进入记忆卡片。</div>')
            table_output = gr.Dataframe(
                label="English / 中文 / Deutsch",
                interactive=False,
                row_count=(6, "dynamic"),
                col_count=(5, "fixed"),
            )

    with gr.Accordion("使用建议 / Troubleshooting", open=False):
        gr.Markdown(
            f"""
            - 当前模型：`{args.model}`；设备：`{args.device}`；图片尺寸：`{args.imgsz}`；置信度阈值：`{args.conf}`。
            - CPU 慢是正常的。弱服务器可运行：`python app.py --device cpu --imgsz 320`。
            - 识别不到时，优先换清晰图片；其次再考虑降低 `--conf`。
            - 未收录词会写入 `unknown_words.tsv`，后续可以手动补充中文和德语。
            - 在服务器上给局域网访问：`python app.py --host 0.0.0.0 --port 7860`。
            """
        )

    gr.HTML('<div class="pic-footer">PIC-CV-LAN · Local-first vocabulary capture tool · Built for language learning, not API burning.</div>')

    run_button.click(
        fn=detect_and_translate,
        inputs=image_input,
        outputs=[annotated_output, status_output, table_output, file_output],
    )


if __name__ == "__main__":
    demo.queue().launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )
