import shutil
import threading
import webbrowser
from pathlib import Path

import emoji
import gradio as gr
import numpy as np
import soundfile as sf  # Used for reading and cutting audio files
import torch
import torchaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# Import internationalization module
from utils.translator import _, initialize_from_main

if __name__ == "__main__":
    # Initialize language system
    initialize_from_main()

# VAD model path
vad_model_dir = "fsmn-vad"

# ASR model paths - Whisper default used
asr_models = {
    "SenseVoiceSmall": "iic/SenseVoiceSmall",
    "Whisper": "iic/Whisper-large-v3-turbo",
    "Paraformer": "paraformer-zh",
}

# Language options
language_options = {
    "SenseVoiceSmall": ["auto", "zh", "en", "yue", "ja", "ko"],
    "Whisper": ["auto", "zh", "en", "ja"],
    "Paraformer": ["auto", "zh"],
}

# Loaded model cache
loaded_models = {}


def get_model(model_name):
    if model_name not in loaded_models:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if model_name == "SenseVoiceSmall":
            loaded_models[model_name] = AutoModel(
                model=asr_models[model_name],
                device=device,
                disable_update=True,
            )
        elif model_name == "Whisper":
            loaded_models[model_name] = AutoModel(
                model=asr_models[model_name],
                device=device,
                vad_kwargs={"max_single_segment_time": 30000},
                disable_update=True,
            )
        elif model_name == "Paraformer":
            loaded_models[model_name] = AutoModel(
                model=asr_models[model_name],
                device=device,
                disable_update=True,
            )
    return loaded_models[model_name]


def open_page():
    webbrowser.open_new_tab("http://127.0.0.1:7860")


# Define timestamp format
def reformat_time(second):
    hours = int(second // 3600)
    minutes = int((second % 3600) // 60)
    secs = int(second % 60)
    millis = int((second % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# Convert speech recognition text to SRT subtitle file
def write_srt(srt_result, srt_file):
    with Path.open(srt_file, "w", encoding="utf-8") as f:
        f.writelines(srt_result)


# 将语音识别得到的文本转写为TXT subtitle file
def write_txt(txt_result, txt_file):
    with Path.open(txt_file, "w", encoding="utf-8") as f:
        f.writelines(txt_result)


def cut_wav_to_ndarray(wav_path: str, start_s: float, end_s: float) -> np.ndarray:
    if end_s <= start_s:
        raise ValueError("end_s must be > start_s")

    with sf.SoundFile(wav_path) as f:
        sr = f.samplerate
        channels = f.channels
        start_frame = max(0, int(start_s / 1000 * sr))
        end_frame = max(start_frame + 1, int(end_s / 1000 * sr))
        frames = end_frame - start_frame
        f.seek(start_frame)
        audio = f.read(frames, dtype="float32")  # read as float32
    # if stereo, convert to mono
    if channels > 1:
        audio = np.mean(audio, axis=1)  # take average
    if sr != 16000:
        # convert to tensor, resample, convert back NumPy
        audio_tensor = torch.tensor(audio)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio_tensor).numpy()
    return audio


# model inference function
def model_inference(input_wav, model_name, language, silence_threshold):
    srt_file = Path(input_wav).with_suffix(".srt")
    txt_file = Path(input_wav).with_suffix(".txt")

    asr_model = get_model(model_name)

    selected_language = None
    if model_name == "SenseVoiceSmall":
        language_abbr = {
            "auto": "auto",
            "zh": "zh",
            "en": "en",
            "yue": "yue",
            "ja": "ja",
            "ko": "ko",
            "nospeech": "nospeech",
        }
        selected_language = language_abbr.get(language, "auto")
    elif model_name == "Whisper":
        language_abbr = {
            "auto": None,
            "zh": "zh",
            "en": "en",
            "ja": "ja",
        }
        selected_language = language_abbr.get(language)
    elif model_name == "Paraformer":
        language_abbr = {
            "auto": "zh",
            "zh": "zh",
        }
        selected_language = language_abbr.get(language, "zh")

    # load VAD model
    vad_model = AutoModel(
        model=vad_model_dir,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        disable_update=True,
        max_end_silence_time=silence_threshold,
    )

    # use VAD model to process audio file
    vad_res = vad_model.generate(
        input=input_wav,
        cache={},
        max_single_segment_time=30000,
    )

    segments = vad_res[0]["value"]

    srt_result = ""
    txt_result = []
    data = []
    srt_id = 1

    for segment in segments:
        start_time, end_time = segment
        audio_temp = cut_wav_to_ndarray(input_wav, start_time, end_time)

        cleaned_text = ""
        if model_name == "SenseVoiceSmall":
            res = asr_model.generate(
                input=audio_temp,
                cache={},
                language=selected_language,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
                ban_emo_unk=False,
            )
            cleaned_text = rich_transcription_postprocess(res[0]["text"])
            cleaned_text = emoji.replace_emoji(cleaned_text, replace="")
            if selected_language not in ["en", "ko"]:
                cleaned_text = cleaned_text.replace(" ", "").strip()
        elif model_name == "Whisper":
            prompt_dict = {
                "auto": "",
                "en": "Tom, There is a Chinese person among them.",
                "zh": "我是一个台湾人，也是一个中国人。",
                "ja": "その中に、一人の日本人がいます。誰だと思いますか？",
            }
            decodingoptions = {
                "task": "transcribe",
                "language": selected_language,
                "beam_size": None,
                "fp16": True,
                "without_timestamps": True,
                "prompt": prompt_dict.get(language, ""),
            }
            res = asr_model.generate(
                input=audio_temp,
                DecodingOptions=decodingoptions,
                batch_size_s=0,
            )
            cleaned_text = res[0]["text"]
        elif model_name == "Paraformer":
            res = asr_model.generate(
                input=audio_temp,
                punc_model="ct-punc-c",
                batch_size_s=300,
                hotword="",
            )
            cleaned_text = res[0]["text"]

        srt_result += (
            str(srt_id)
            + "\n"
            + str(reformat_time(start_time / 1000))
            + " --> "
            + str(reformat_time(end_time / 1000))
            + "\n"
            + cleaned_text
            + "\n\n"
        )
        txt_result.append(cleaned_text)
        data.append([str(srt_id), reformat_time(start_time / 1000), reformat_time(end_time / 1000), cleaned_text])
        srt_id += 1

    write_srt(srt_result, srt_file)
    write_txt(txt_result, txt_file)
    gr.Info(_("transcript_completed", filename=Path(input_wav).name))
    return data


# save subtitle files to selected folder
def save_file(audio_inputs, path_input_text):
    if not Path(path_input_text).is_dir() or path_input_text.strip() == "":
        gr.Warning(_("invalid_folder"))
    else:
        try:
            srt_file = Path(audio_inputs).with_suffix(".srt")
            txt_file = Path(audio_inputs).with_suffix(".txt")
            shutil.copy2(srt_file, path_input_text)
            shutil.copy2(txt_file, path_input_text)
            gr.Info(_("files_saved", srt_file=srt_file.name, txt_file=txt_file.name))
        except Exception as e:
            gr.Warning(_("save_error", error=e))


# multi-file transcription
def multi_file_asr(multi_files_upload, model_name, language, silence_threshold):
    num = 0
    for audio_inputs in multi_files_upload:
        model_inference(audio_inputs, model_name, language, silence_threshold)
        num += 1
    gr.Info(_("total_transcriptions", num=num))


# save subtitle files to selected folder
def save_multi_srt(multi_files_upload, path_input_text):
    for audio_inputs in multi_files_upload:
        save_file(audio_inputs, path_input_text)


def get_html_content():
    return f"""
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">{_("title")}</h2>
    <p style="font-size: 18px;margin-left: 20px;">{_("description")}</p>
    <p style="margin-left: 20px;"><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">{_("github_links")}</a>
    <a href="https://github.com/jianchang512/sense-api" target="_blank">{_("sense_api_repo")}</a>
</div>
"""


def update_language_options(model_name):
    return gr.Dropdown(choices=language_options[model_name], value="auto", label=_("spoken_language"))


def launch():
    # translator initialized on module import

    with gr.Blocks(title="SenseVoice & Whisper WebUI") as demo:
        # page title and introduction
        gr.HTML(get_html_content())

        with gr.Column():
            model_selector = gr.Radio(
                choices=["SenseVoiceSmall", "Whisper", "Paraformer"],
                value="SenseVoiceSmall",
                label=_("select_model"),
            )
            with gr.Accordion(_("configuration")), gr.Row():
                language_inputs = gr.Dropdown(
                    choices=language_options["SenseVoiceSmall"], value="auto", label=_("spoken_language")
                )
                end_silence_time = gr.Slider(
                    label=_("silence_threshold"), minimum=0, maximum=6000, step=50, value=800, interactive=True
                )
        with gr.Tab(label=_("single_file_transcription")), gr.Column():
            audio_inputs = gr.Audio(label=_("upload_audio"), type="filepath")

            with gr.Row():
                stre_btn = gr.Button(_("start_transcription"), variant="primary")
                save_btn = gr.Button(_("save_subtitles"), variant="primary")
            path_input_text = gr.Text(label=_("save_path"), interactive=True, placeholder=_("save_path_placeholder"))
            output_table = gr.Dataframe(
                headers=[
                    _("table_headers.no"),
                    _("table_headers.start_time"),
                    _("table_headers.end_time"),
                    _("table_headers.subtitle_content"),
                ],
                label=_("transcription_results"),
            )

        model_selector.change(update_language_options, inputs=model_selector, outputs=language_inputs)

        stre_btn.click(
            model_inference,
            inputs=[audio_inputs, model_selector, language_inputs, end_silence_time],
            outputs=output_table,
        )

        save_btn.click(save_file, inputs=[audio_inputs, path_input_text], outputs=[])

        with gr.Tab(label=_("multi_file_transcription")), gr.Column():
            multi_files_upload = gr.File(
                label=_("upload_audio_files"),
                file_count="directory",
                file_types=[".mp3", ".wav", ".flac", ".m4a", ".ogg"],
            )
            with gr.Row():
                stre_btn_multi = gr.Button(_("start_transcription"), variant="primary")
                save_btn_multi = gr.Button(_("save_subtitles"), variant="primary")
            path_input_text_multi = gr.Text(
                label=_("save_path"), interactive=True, placeholder=_("save_path_placeholder")
            )

        model_selector.change(update_language_options, inputs=model_selector, outputs=language_inputs)

        stre_btn_multi.click(
            multi_file_asr,
            inputs=[multi_files_upload, model_selector, language_inputs, end_silence_time],
            outputs=[],
        )

        save_btn_multi.click(save_multi_srt, inputs=[multi_files_upload, path_input_text_multi], outputs=[])

    threading.Thread(target=open_page).start()
    demo.launch()


if __name__ == "__main__":
    launch()
