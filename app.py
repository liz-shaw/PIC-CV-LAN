from __future__ import annotations

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

# Load once when the app starts.
detector = ObjectDetector(model_name="yolo11n.pt", confidence_threshold=0.35)
vocab_store = VocabStore(VOCAB_PATH)


def detect_and_translate(image: Image.Image | None):
    if image is None:
        empty_df = pd.DataFrame(columns=["English", "中文", "Deutsch", "Confidence", "Source"])
        return None, empty_df, None, "请先上传或拍摄一张图片。"

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
                "Source": "YOLO",
            }
        )

    if rows:
        output_file = export_rows_to_tsv(rows, OUTPUT_PATH)
        message = f"识别到 {len(rows)} 个物品。"
        if unknown_count:
            message += f" 其中 {unknown_count} 个词未收录，已写入 unknown_words.tsv。"
    else:
        output_file = None
        message = "没有识别到置信度足够高的常见物品。可以换一张更清晰的图片，或降低阈值。"

    df = pd.DataFrame(rows, columns=["English", "中文", "Deutsch", "Confidence", "Source"])
    return annotated_image, df, str(output_file) if output_file else None, message


with gr.Blocks(title="PIC-CV-LAN") as demo:
    gr.Markdown(
        """
        # PIC-CV-LAN
        本地图片物品识别 → English / 中文 / Deutsch → TSV 导出。  
        不需要 AI token；首次运行需要下载 YOLO 模型文件。
        """
    )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            sources=["upload", "webcam"],
            label="上传图片或使用摄像头",
        )
        annotated_output = gr.Image(label="检测标注图")

    run_button = gr.Button("识别并生成三语词条", variant="primary")
    status_output = gr.Textbox(label="状态", interactive=False)
    table_output = gr.Dataframe(label="识别结果", interactive=False)
    file_output = gr.File(label="下载 TSV")

    run_button.click(
        fn=detect_and_translate,
        inputs=image_input,
        outputs=[annotated_output, table_output, file_output, status_output],
    )


if __name__ == "__main__":
    demo.launch()
