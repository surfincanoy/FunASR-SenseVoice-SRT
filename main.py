import shutil
import string
import threading
import webbrowser
from pathlib import Path

import emoji
import gradio as gr
import numpy as np
import soundfile as sf
import torch
from fireredvad import FireRedVad, FireRedVadConfig
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from pydub import AudioSegment
from scipy.signal import resample as scipy_resample
from utils.translator import _, initialize_from_main, translator

if __name__ == "__main__":
    initialize_from_main()


def remove_trailing_punctuation(text: str) -> str:
    trailing_punct = string.punctuation + "。，、；：！？.?!"
    text = text.rstrip(trailing_punct)
    text = text.rstrip() + " "
    return text


def convert_to_wav(input_path: str, output_path: Path) -> str:
    input_path_obj = Path(input_path)
    if input_path_obj.suffix.lower() == ".wav":
        try:
            with sf.SoundFile(input_path) as f:
                if f.samplerate == 16000 and f.channels == 1:
                    return input_path  # 直接使用原文件
        except Exception:
            pass

    # 需要转换时，生成临时文件
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(str(output_path), format="wav")
    return str(output_path)


# ASR model paths - prioritize local models
LOCAL_MODELS_DIR = Path(".")
FUN_AUDIO_LLM_DIR = LOCAL_MODELS_DIR / "FunAudioLLM"
IIC_DIR = LOCAL_MODELS_DIR / "iic"


def get_local_model_path(relative_path: str) -> str:
    """Check if local model exists, return local path or HuggingFace ID"""
    local_path = LOCAL_MODELS_DIR / relative_path
    if local_path.exists():
        return str(local_path)
    return relative_path


asr_models = {
    "SenseVoiceSmall": get_local_model_path("iic/SenseVoiceSmall"),
    "Whisper": get_local_model_path("iic/Whisper-large-v3-turbo"),
    "Paraformer-zh": "paraformer-zh",
    "Paraformer-en": get_local_model_path(
        "iic/speech_paraformer_asr-en-16k-vocab4199-pytorch"
    ),
    "Fun-ASR-Nano": get_local_model_path("FunAudioLLM/Fun-ASR-Nano-2512"),
    "Fun-ASR-MLT-Nano": get_local_model_path("FunAudioLLM/Fun-ASR-MLT-Nano-2512"),
}

# Language options: {display_key: asr_value}
# display_key is looked up in locale language_names for GUI display
# asr_value is the actual value passed to the ASR model
language_options = {
    "SenseVoiceSmall": {
        "zh": "zh",
        "en": "en",
        "yue": "yue",
        "ja": "ja",
        "ko": "ko",
    },
    "Whisper": {
        "zh": "zh",
        "en": "en",
        "ja": "ja",
    },
    "Paraformer-zh": {
        "zh": "zh",
    },
    "Paraformer-en": {
        "en": "en",
    },
    "Fun-ASR-Nano": {
        "中文": "中文",
        "英文": "英文",
        "日文": "日文",
    },
    "Fun-ASR-MLT-Nano": {
        "中文": "中文",
        "英文": "英文",
        "日文": "日文",
        "粤语": "粤语",
        "韩文": "韩文",
        "越南语": "越南语",
        "印尼语": "印尼语",
        "泰语": "泰语",
        "马来语": "马来语",
        "菲律宾语": "菲律宾语",
        "阿拉伯语": "阿拉伯语",
        "印地语": "印地语",
        "保加利亚语": "保加利亚语",
        "克罗地亚语": "克罗地亚语",
        "捷克语": "捷克语",
        "丹麦语": "丹麦语",
        "荷兰语": "荷兰语",
        "爱沙尼亚语": "爱沙尼亚语",
        "芬兰语": "芬兰语",
        "希腊语": "希腊语",
        "匈牙利语": "匈牙利语",
        "爱尔兰语": "爱尔兰语",
        "拉脱维亚语": "拉脱维亚语",
        "立陶宛语": "立陶宛语",
        "马耳他语": "马耳他语",
        "波兰语": "波兰语",
        "葡萄牙语": "葡萄牙语",
        "罗马尼亚语": "罗马尼亚语",
        "斯洛伐克语": "斯洛伐克语",
        "斯洛文尼亚语": "斯洛文尼亚语",
        "瑞典语": "瑞典语",
    },
}

# Default language for each model
default_languages = {
    "SenseVoiceSmall": "en",
    "Whisper": "en",
    "Paraformer-zh": "zh",
    "Paraformer-en": "en",
    "Fun-ASR-Nano": "中文",
    "Fun-ASR-MLT-Nano": "中文",
}

# Loaded model cache
loaded_models = {}


def get_asr_model(model_name, device):
    if model_name not in loaded_models:
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
        elif model_name in ["Paraformer-zh", "Paraformer-en"]:
            loaded_models[model_name] = AutoModel(
                model=asr_models[model_name],
                model_revision="v2.0.4",
                vad_model="fsmn-vad",
                vad_model_revision="v2.0.4",
                punc_model="ct-punc-c",
                punc_model_revision="v2.0.4",
                device=device,
                disable_update=True,
            )
        elif model_name in ["Fun-ASR-Nano", "Fun-ASR-MLT-Nano"]:
            loaded_models[model_name] = AutoModel(
                model=asr_models[model_name],
                trust_remote_code=True,
                remote_code="./model.py",
                device=device,
                hub="ms",
                disable_update=True,
            )
    return loaded_models[model_name]


def get_vad_model(
    smooth_window_size,
    speech_threshold,
    min_speech_frame,
    max_speech_frame,
    min_silence_frame,
    merge_silence_frame,
    extend_speech_frame,
):
    vad_config = FireRedVadConfig(
        use_gpu=False,
        smooth_window_size=smooth_window_size,
        speech_threshold=speech_threshold,
        min_speech_frame=min_speech_frame,
        max_speech_frame=max_speech_frame,
        min_silence_frame=min_silence_frame,
        merge_silence_frame=merge_silence_frame,
        extend_speech_frame=extend_speech_frame,
        chunk_max_frame=30000,
    )
    vad = FireRedVad.from_pretrained(
        "./FireRedVAD",
        vad_config,
    )
    return vad


def open_page():
    webbrowser.open_new_tab("http://127.0.0.1:7860")


def reformat_time(second):
    hours = int(second // 3600)
    minutes = int((second % 3600) // 60)
    secs = int(second % 60)
    millis = int((second % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(srt_result, srt_file):
    with Path.open(srt_file, "w", encoding="utf-8") as f:
        f.writelines(srt_result)


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
        audio = f.read(frames, dtype="float32")
    if channels > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        num_samples = int(len(audio) * 16000 / sr)
        audio = scipy_resample(audio, num_samples)
    return audio


def extract_segment_tensor(
    waveform: torch.Tensor, sample_rate: int, start: float, end: float
) -> torch.Tensor:
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    return waveform[start_sample:end_sample]


def model_inference(
    input_wav,
    model_name,
    language,
    remove_trailing_punct,
    smooth_window_size,
    speech_threshold,
    min_speech_frame,
    max_speech_frame,
    min_silence_frame,
    merge_silence_frame,
    extend_speech_frame,
):
    srt_file = Path(input_wav).with_suffix(".srt")
    txt_file = Path(input_wav).with_suffix(".txt")

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    asr_model = get_asr_model(model_name, device)

    selected_language = language_options[model_name].get(
        language, default_languages[model_name]
    )

    vad = get_vad_model(
        smooth_window_size,
        speech_threshold,
        min_speech_frame,
        max_speech_frame,
        min_silence_frame,
        merge_silence_frame,
        extend_speech_frame,
    )

    wav_path = Path(input_wav).with_suffix(".wav")
    wav_file = convert_to_wav(input_wav, wav_path)

    if model_name in ["Fun-ASR-Nano", "Fun-ASR-MLT-Nano"]:
        waveform_np, sample_rate = sf.read(wav_file, dtype="float32")
        if waveform_np.ndim > 1:
            waveform_np = np.mean(waveform_np, axis=1)
        waveform = torch.from_numpy(waveform_np)
        result, probs = vad.detect(wav_file)
        timestamps = result.get("timestamps", [])

        srt_result = ""
        txt_lines = []
        data = []
        srt_id = 1

        for idx, (start, end) in enumerate(timestamps):
            segment_tensor = extract_segment_tensor(waveform, sample_rate, start, end)

            res = asr_model.generate(
                input=[segment_tensor],
                cache={},
                batch_size=1,
                language=selected_language,
                itn=True,
            )
            text = res[0]["text"]
            if remove_trailing_punct:
                text = remove_trailing_punctuation(text)
            if text.strip():
                start_ms = start * 1000
                end_ms = end * 1000
                srt_result += (
                    str(srt_id)
                    + "\n"
                    + reformat_time(start)
                    + " --> "
                    + reformat_time(end)
                    + "\n"
                    + text
                    + "\n\n"
                )
                txt_lines.append(text)
                data.append(
                    [
                        str(srt_id),
                        reformat_time(start),
                        reformat_time(end),
                        text,
                    ]
                )
                srt_id += 1

        del waveform
    else:
        result, probs = vad.detect(wav_file)
        timestamps = result.get("timestamps", [])

        srt_result = ""
        txt_lines = []
        data = []
        srt_id = 1

        for start, end in timestamps:
            start_ms = start * 1000
            end_ms = end * 1000
            audio_temp = cut_wav_to_ndarray(input_wav, start_ms, end_ms)

            cleaned_text = ""
            if model_name == "SenseVoiceSmall":
                res = asr_model.generate(
                    input=audio_temp,
                    cache={},
                    param_dict={"use_punc": True},
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
                if remove_trailing_punct:
                    cleaned_text = remove_trailing_punctuation(cleaned_text)
            elif model_name == "Whisper":
                prompt_dict = {
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
                if remove_trailing_punct:
                    cleaned_text = remove_trailing_punctuation(cleaned_text)
            elif model_name in ["Paraformer-zh", "Paraformer-en"]:
                res = asr_model.generate(
                    input=audio_temp,
                    batch_size_s=300,
                    hotword="",
                )
                cleaned_text = res[0]["text"]
                if remove_trailing_punct:
                    cleaned_text = remove_trailing_punctuation(cleaned_text)
            elif model_name == "Whisper":
                prompt_dict = {
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
                cleaned_text = remove_trailing_punctuation(cleaned_text)
            elif model_name in ["Paraformer-zh", "Paraformer-en"]:
                res = asr_model.generate(
                    input=audio_temp,
                    batch_size_s=300,
                    hotword="",
                )
                cleaned_text = res[0]["text"]
                cleaned_text = remove_trailing_punctuation(cleaned_text)

            srt_result += (
                str(srt_id)
                + "\n"
                + reformat_time(start)
                + " --> "
                + reformat_time(end)
                + "\n"
                + cleaned_text
                + "\n\n"
            )
            txt_lines.append(cleaned_text)
            data.append(
                [
                    str(srt_id),
                    reformat_time(start),
                    reformat_time(end),
                    cleaned_text,
                ]
            )
            srt_id += 1

    write_srt(srt_result, srt_file)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("".join(txt_lines))
    gr.Info(_("transcript_completed", filename=Path(input_wav).name))
    return data


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


def multi_file_asr(
    multi_files_upload,
    model_name,
    language,
    remove_trailing_punct,
    smooth_window_size,
    speech_threshold,
    min_speech_frame,
    max_speech_frame,
    min_silence_frame,
    merge_silence_frame,
    extend_speech_frame,
):
    num = 0
    for audio_inputs in multi_files_upload:
        model_inference(
            audio_inputs,
            model_name,
            language,
            remove_trailing_punct,
            smooth_window_size,
            speech_threshold,
            min_speech_frame,
            max_speech_frame,
            min_silence_frame,
            merge_silence_frame,
            extend_speech_frame,
        )
        num += 1
    gr.Info(_("total_transcriptions", num=num))


def save_multi_srt(multi_files_upload, path_input_text):
    for audio_inputs in multi_files_upload:
        save_file(audio_inputs, path_input_text)


def get_html_content():
    return f"""
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">{"FireRed VAD + ASR SRT Generator"}</h2>
    <p style="font-size: 18px;margin-left: 20px;">{"Integrated FireRedVAD with SenseVoice/Whisper/Paraformer/Fun-ASR for subtitle generation"}</p>
    <p style="margin-left: 20px;"><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">SenseVoice</a>
    <a href="https://github.com/FunAudioLLM/Fun-ASR" target="_blank">Fun-ASR</a>
    <a href="https://github.com/FireRedTeam/FireRedVAD" target="_blank">FireRedVAD</a>
</div>
"""


def update_language_options(model_name):
    choices = []
    for display_key, asr_value in language_options[model_name].items():
        translated = translator.t("language_names." + display_key)
        if translated == "language_names." + display_key:
            translated = display_key
        choices.append((translated, display_key))
    return gr.Dropdown(
        choices=choices,
        value=default_languages[model_name],
        label=_("spoken_language"),
    )


def launch():
    with gr.Blocks(
        title="FunASR&SenseVoice SRT Generator",
        theme=gr.themes.Default(primary_hue="green", secondary_hue="blue"),
    ) as demo:
        gr.HTML(get_html_content())

        with gr.Column():
            model_selector = gr.Radio(
                choices=[
                    "SenseVoiceSmall",
                    "Whisper",
                    "Paraformer-zh",
                    "Paraformer-en",
                    "Fun-ASR-Nano",
                    "Fun-ASR-MLT-Nano",
                ],
                value="SenseVoiceSmall",
                label=_("select_model"),
            )
            with gr.Accordion(_("configuration"), open=True), gr.Row():
                language_inputs = gr.Dropdown(
                    choices=[
                        (translator.t("language_names." + k), k)
                        for k in language_options["SenseVoiceSmall"]
                    ],
                    value=default_languages["SenseVoiceSmall"],
                    label=_("spoken_language"),
                )
                remove_trailing_punct = gr.Checkbox(
                    value=False,
                    label=_("remove_trailing_punct"),
                    info=_("remove_trailing_punct_info"),
                )

        with gr.Accordion(_("fireredvad_params"), open=False):
            with gr.Row():
                vad_smooth_window_size = gr.Slider(
                    label=_("vad_smooth_window_size"),
                    minimum=1,
                    maximum=21,
                    step=1,
                    value=5,
                    interactive=True,
                    info=_("vad_smooth_window_size_info"),
                )
                vad_speech_threshold = gr.Slider(
                    label=_("vad_speech_threshold"),
                    minimum=0.1,
                    maximum=0.9,
                    step=0.05,
                    value=0.4,
                    interactive=True,
                    info=_("vad_speech_threshold_info"),
                )
            with gr.Row():
                vad_min_speech_frame = gr.Slider(
                    label=_("vad_min_speech_frame"),
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=20,
                    interactive=True,
                    info=_("vad_min_speech_frame_info"),
                )
                vad_max_speech_frame = gr.Slider(
                    label=_("vad_max_speech_frame"),
                    minimum=100,
                    maximum=5000,
                    step=50,
                    value=2000,
                    interactive=True,
                    info=_("vad_max_speech_frame_info"),
                )
                vad_min_silence_frame = gr.Slider(
                    label=_("vad_min_silence_frame"),
                    minimum=1,
                    maximum=200,
                    step=1,
                    value=100,
                    interactive=True,
                    info=_("vad_min_silence_frame_info"),
                )
            with gr.Row():
                vad_merge_silence_frame = gr.Slider(
                    label=_("vad_merge_silence_frame"),
                    minimum=0,
                    maximum=50,
                    step=1,
                    value=0,
                    interactive=True,
                    info=_("vad_merge_silence_frame_info"),
                )
                vad_extend_speech_frame = gr.Slider(
                    label=_("vad_extend_speech_frame"),
                    minimum=0,
                    maximum=50,
                    step=1,
                    value=0,
                    interactive=True,
                    info=_("vad_extend_speech_frame_info"),
                )

        with gr.Tab(label=_("single_file_transcription")), gr.Column():
            audio_inputs = gr.Audio(label=_("upload_audio"), type="filepath")

            with gr.Row():
                stre_btn = gr.Button(_("start_transcription"), variant="primary")
                save_btn = gr.Button(_("save_subtitles"), variant="primary")
            path_input_text = gr.Text(
                label=_("save_path"),
                interactive=True,
                placeholder=_("save_path_placeholder"),
            )
            output_table = gr.Dataframe(
                headers=[
                    _("table_headers.no"),
                    _("table_headers.start_time"),
                    _("table_headers.end_time"),
                    _("table_headers.subtitle_content"),
                ],
                label=_("transcription_results"),
            )

        model_selector.change(
            update_language_options,
            inputs=model_selector,
            outputs=language_inputs,
        )

        stre_btn.click(
            model_inference,
            inputs=[
                audio_inputs,
                model_selector,
                language_inputs,
                remove_trailing_punct,
                vad_smooth_window_size,
                vad_speech_threshold,
                vad_min_speech_frame,
                vad_max_speech_frame,
                vad_min_silence_frame,
                vad_merge_silence_frame,
                vad_extend_speech_frame,
            ],
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
                label=_("save_path"),
                interactive=True,
                placeholder=_("save_path_placeholder"),
            )

        model_selector.change(
            update_language_options,
            inputs=model_selector,
            outputs=language_inputs,
        )

        stre_btn_multi.click(
            multi_file_asr,
            inputs=[
                multi_files_upload,
                model_selector,
                language_inputs,
                remove_trailing_punct,
                vad_smooth_window_size,
                vad_speech_threshold,
                vad_min_speech_frame,
                vad_max_speech_frame,
                vad_min_silence_frame,
                vad_merge_silence_frame,
                vad_extend_speech_frame,
            ],
            outputs=[],
        )

        save_btn_multi.click(
            save_multi_srt,
            inputs=[multi_files_upload, path_input_text_multi],
            outputs=[],
        )

    threading.Thread(target=open_page).start()
    demo.launch()


if __name__ == "__main__":
    launch()
