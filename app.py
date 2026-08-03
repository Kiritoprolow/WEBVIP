"""
app.py — Dịch & Lồng Tiếng Video Trung -> Việt: điểm vào Gradio (điều phối pipeline + UI).
============================================================================================
PHIÊN BẢN ĐÃ VIẾT LẠI để khớp với các module THẬT hiện có trong dự án (không phải bản gốc
tách 5 file config/downloader/translator/tts_engine/video_editor như docstring cũ mô tả).

ĐỐI CHIẾU VỚI CODE THẬT — những gì được TÁI SỬ DỤNG NGUYÊN VẸN (không sửa 1 dòng logic):
    config.py           - toàn bộ hằng số, client Groq/Gemini, cache utils, run_subprocess
    downloader.py        - download_video_from_url, extract_clean_url, validate_video
    translate_and_tts.py - translate_script (dịch Gemini theo lô), _tts_generate_all_lines
                            (sinh audio edge-tts song song), format_srt_timestamp, _run_async,
                            CONTENT_STYLES/DEFAULT_CONTENT_STYLE
    video_editor.py      - build_canvas_filter (canvas 3 lớp blur+flip+speed), mix_audio
                            (trộn TTS + audio gốc), merge_video_audio (ghép video+audio cuối)

NHỮNG GÌ LÀ CODE MỚI (viết thêm trong chính file này, vì KHÔNG tồn tại ở đâu khác trong dự
án — đã hỏi ý kiến và được đồng ý viết mới cho phần này):
    _extract_audio_for_stt      - tách audio bằng ffmpeg để đưa vào Whisper
    _transcribe_audio_groq      - gọi Groq Whisper (client đã có sẵn trong config.py) lấy
                                   segment kèm timestamp gốc của video
    _build_srt_from_segments    - dựng .srt theo ĐÚNG timestamp gốc (khác hẳn cách
                                   translate_and_tts.py tự dựng timeline mới theo độ dài TTS)
    _build_force_style          - chuyển ASS_STYLE_PRESETS (config.py) thành chuỗi
                                   force_style cho bộ lọc `subtitles` gốc của FFmpeg
    _render_video_only           - Hybrid Dual-Engine: thử GPU Fast Track trước
                                   (_render_video_only_gpu, @spaces.GPU + NVENC), hết quota
                                   ZeroGPU hoặc lỗi gì cũng tự rớt về CPU (libx264). Cả 2
                                   nhánh đều ghép build_canvas_filter() thật + bộ lọc
                                   `subtitles` của FFmpeg (đọc thẳng .srt, không cần .ass)
    _extract_speed_corrected_audio / _speed_correct_audio
                                 - đồng bộ audio (gốc hoặc lồng tiếng) với tốc độ video mới
                                   (speed) bằng bộ lọc atempo của FFmpeg
    _build_synced_dub_track      - đặt từng câu audio TTS đúng vào mốc thời gian (start)
                                   của segment gốc trong video (thay vì nối tuần tự như
                                   generate_dubbed_audio_and_srt() của translate_and_tts.py)
    _generate_social_caption     - gọi Gemini (client có sẵn trong config.py) tạo Caption
                                   & Hashtag, có fallback nếu lỗi

TÍNH NĂNG ĐÃ BỎ vì KHÔNG có code thật nào để tái sử dụng, và tự viết mới sẽ rủi ro cao
(chưa test được với hạ tầng thật của bạn — BGM cần tải từ GitHub Release, YouTube cần OAuth
thật, tách 2 tập cần logic cắt segment + resync phức tạp):
    - Smart BGM Fitter (chọn/tải/trộn nhạc nền từ GitHub Release)
    - Tự động đăng YouTube Shorts
    - Tự động tách video dài > 3 phút thành 2 tập
    - Zoom / Saturation / Brightness / Pitch (build_canvas_filter() thật chỉ nhận `speed`,
      không có các tham số này)
Nếu bạn có code thật cho các phần trên, gửi thêm là mình lắp lại được ngay.

Hybrid GPU/CPU render: ĐÃ khôi phục (xem _render_video_only) — thử NVENC qua @spaces.GPU
trước, hết quota ZeroGPU/lỗi gì cũng tự rớt về CPU (libx264), không cần Space đổi hardware.

TÍNH NĂNG MỚI được bổ sung vì có code thật cho phép làm (mix_audio() thật hỗ trợ trộn tỉ lệ
audio gốc, translate_and_tts.py thật hỗ trợ chọn văn phong dịch):
    - Chọn "Văn phong dịch" (drama / comedy / kids) — dùng CONTENT_STYLES thật.
    - Bật "Giữ âm thanh gốc" khi lồng tiếng + chỉnh âm lượng audio gốc (mix_audio()).

Biến môi trường bắt buộc (giống hệt config.py):
    GROQ_API_KEY   - Groq API key (Whisper STT)
    GEMINI_API_KEY - Google Gemini API key (dịch thuật + caption)

Gói hệ thống bắt buộc: ffmpeg, ffprobe (ffmpeg cần build có libass để bộ lọc `subtitles`
hoạt động — bản ffmpeg mặc định trên hầu hết distro/HF Spaces đều có sẵn).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import gradio as gr
import spaces
from pydub import AudioSegment

from config import (
    ASPECT_MODES,
    ASPECT_ORIGINAL,
    ASS_FONT_NAME,
    ASS_FONT_SIZE_DEFAULT,
    ASS_MARGIN_LR_DEFAULT,
    ASS_MARGIN_V_DEFAULT,
    ASS_STYLE_PRESETS,
    CACHED_OUTPUTS_DIR,
    DEFAULT_ASPECT_MODE,
    DEFAULT_SUBTITLE_STYLE_LABEL,
    DEFAULT_VOICE_LABEL,
    GEMINI_MODEL_NAME,
    MAX_DURATION_SEC,
    MAX_FILE_SIZE_MB,
    MODE_SUBTITLE_AND_DUB,
    MODE_SUBTITLE_ONLY,
    PROCESSING_MODES,
    PipelineError,
    Segment,
    SPEED_DEFAULT,
    SPEED_MAX,
    SPEED_MIN,
    TTS_PADDING_DEFAULT,
    TTS_PADDING_MAX,
    TTS_PADDING_MIN,
    TTS_RATE_DEFAULT,
    TTS_RATE_MAX,
    TTS_RATE_MIN,
    VOICE_OPTIONS,
    WHISPER_MODEL_NAME,
    build_cache_key,
    compute_sha256,
    gemini_client,
    get_cached_caption,
    get_cached_output,
    groq_client,
    list_cached_output_names,
    logger,
    run_subprocess,
    save_to_history,
    store_caption_in_cache,
    store_in_cache,
)
from downloader import download_video_from_url, extract_clean_url, validate_video
from translate_and_tts import (
    CONTENT_STYLES,
    DEFAULT_CONTENT_STYLE,
    PipelineError as TTSPipelineError,
    _run_async,
    _tts_generate_all_lines,
    format_srt_timestamp,
    translate_script,
)
from video_editor import build_canvas_filter, merge_video_audio, mix_audio

# --------------------------------------------------------------------------- #
# Hằng số MỚI (chưa tồn tại trong config.py thật vì tính năng "giữ % audio gốc
# khi lồng tiếng" chỉ mới có từ khi video_editor.py thật hỗ trợ mix_audio()).
# --------------------------------------------------------------------------- #
ORIGINAL_VOLUME_MIN, ORIGINAL_VOLUME_MAX, ORIGINAL_VOLUME_DEFAULT = 0.0, 1.0, 0.2
CAPTION_FALLBACK_TEXT = "🎬 Video hấp dẫn, đừng bỏ lỡ!\n#trending #viral #xuhuong #fyp"


# --------------------------------------------------------------------------- #
# CODE MỚI — Bước 1: Tách audio + STT (Groq Whisper) lấy segment kèm timestamp gốc
# --------------------------------------------------------------------------- #

def _extract_audio_for_stt(video_path: Path, output_path: Path) -> None:
    """Tách audio mono 16kHz WAV — định dạng khuyến nghị cho Whisper API."""
    run_subprocess(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
            str(output_path),
        ],
        step_name="Tách âm thanh (STT)",
    )


def _transcribe_audio_groq(audio_path: Path) -> list[Segment]:
    """
    Gọi Groq Whisper (groq_client + WHISPER_MODEL_NAME đã được khởi tạo sẵn trong
    config.py — chỉ chưa có hàm nào gọi tới) lấy transcript kèm timestamp từng câu
    (response_format="verbose_json" trả về field `segments`).
    """
    with open(audio_path, "rb") as f:
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_path.name, f.read()),
            model=WHISPER_MODEL_NAME,
            language="zh",
            response_format="verbose_json",
        )

    raw_segments = getattr(transcription, "segments", None) or []
    segments: list[Segment] = []
    for i, raw in enumerate(raw_segments):
        text = (raw.get("text") if isinstance(raw, dict) else getattr(raw, "text", "")) or ""
        text = text.strip()
        if not text:
            continue
        start = raw.get("start") if isinstance(raw, dict) else getattr(raw, "start", 0.0)
        end = raw.get("end") if isinstance(raw, dict) else getattr(raw, "end", 0.0)
        segments.append(Segment(index=i, start=float(start), end=float(end), text_zh=text))

    if not segments:
        raise PipelineError("Groq Whisper không nhận diện được câu thoại nào trong video.")

    logger.info("STT hoàn tất: %d segment.", len(segments))
    return segments


# --------------------------------------------------------------------------- #
# CODE MỚI — Bước 2: Dựng .srt theo ĐÚNG timestamp gốc của video
# --------------------------------------------------------------------------- #

def _build_srt_from_segments(segments: list[Segment]) -> str:
    """
    Khác với generate_dubbed_audio_and_srt() thật (translate_and_tts.py) — hàm đó dựng
    timestamp MỚI theo độ dài audio TTS. Ở đây timestamp giữ NGUYÊN theo video gốc, vì
    mục tiêu là gắn phụ đề khớp đúng khung hình đang chiếu, không phải khớp audio TTS.
    """
    blocks: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start_ts = format_srt_timestamp(int(seg.start * 1000))
        end_ts = format_srt_timestamp(int(seg.end * 1000))
        blocks.append(f"{i}\n{start_ts} --> {end_ts}\n{seg.text_vi}\n")
    return "\n".join(blocks) + "\n"


# --------------------------------------------------------------------------- #
# CODE MỚI — Bước 3: Render video (canvas thật + gắn cứng phụ đề bằng FFmpeg)
# --------------------------------------------------------------------------- #

def _build_force_style(style_label: str) -> str:
    """Chuyển 1 preset trong ASS_STYLE_PRESETS (config.py, dữ liệu THẬT) thành chuỗi
    force_style cho bộ lọc `subtitles` gốc của FFmpeg (dùng chung engine libass với .ass)."""
    preset = ASS_STYLE_PRESETS.get(style_label, ASS_STYLE_PRESETS[DEFAULT_SUBTITLE_STYLE_LABEL])
    parts = [
        f"FontName={ASS_FONT_NAME}",
        f"FontSize={ASS_FONT_SIZE_DEFAULT}",
        f"PrimaryColour={preset['primary_colour']}",
        f"BackColour={preset['back_colour']}",
        f"OutlineColour={preset['outline_colour']}",
        f"Bold={preset['bold']}",
        f"BorderStyle={preset['border_style']}",
        f"Outline={preset['outline']}",
        f"Shadow={preset['shadow']}",
        f"MarginV={ASS_MARGIN_V_DEFAULT}",
        f"MarginL={ASS_MARGIN_LR_DEFAULT}",
        f"MarginR={ASS_MARGIN_LR_DEFAULT}",
    ]
    return ",".join(parts)


def _build_render_filter_complex(
    srt_path: Path, style_label: str, aspect_mode: str, speed: float
) -> str:
    """Dựng chuỗi filter_complex dùng chung cho cả 2 nhánh GPU/CPU bên dưới."""
    escaped_srt = str(srt_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    force_style = _build_force_style(style_label)

    if aspect_mode == ASPECT_ORIGINAL:
        pts_factor = 1 / speed
        video_chain = f"[0:v]setpts={pts_factor:.6f}*PTS[vbase]"
    else:
        # build_canvas_filter() thật (video_editor.py) — chỉ đổi tên nhãn output cuối
        # [vout] -> [vbase] để nối tiếp bộ lọc subtitles bên dưới.
        video_chain = build_canvas_filter(speed=speed).replace("[vout]", "[vbase]")

    return f"{video_chain};[vbase]subtitles='{escaped_srt}':force_style='{force_style}'[vfinal]"


def _run_ffmpeg_render(
    video_path: Path,
    srt_path: Path,
    style_label: str,
    aspect_mode: str,
    speed: float,
    output_path: Path,
    use_gpu: bool,
) -> None:
    """Chạy FFmpeg render video KHÔNG audio (-an), chọn encoder theo use_gpu."""
    filter_complex = _build_render_filter_complex(srt_path, style_label, aspect_mode, speed)
    codec_args = (
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
        if use_gpu
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter_complex", filter_complex,
        "-map", "[vfinal]",
        "-an",
        *codec_args,
        str(output_path),
    ]
    step_name = "Render video GPU (NVENC)" if use_gpu else "Render video CPU (libx264)"
    run_subprocess(cmd, step_name=step_name)


@spaces.GPU(duration=90)
def _render_video_only_gpu(
    video_path: Path, srt_path: Path, style_label: str, aspect_mode: str, speed: float, output_path: Path
) -> None:
    """
    GPU Fast Track: ZeroGPU cấp phát GPU tạm thời (tối đa 90s) cho riêng lệnh gọi này,
    dùng encoder NVENC (h264_nvenc) để render nhanh hơn CPU. Nếu ffmpeg build không có
    NVENC, hoặc ZeroGPU hết quota, hàm này sẽ raise exception -> để _render_video_only()
    bắt và tự rớt về CPU Graceful Fallback bên dưới.
    """
    _run_ffmpeg_render(video_path, srt_path, style_label, aspect_mode, speed, output_path, use_gpu=True)


def _render_video_only(
    video_path: Path,
    srt_path: Path,
    style_label: str,
    aspect_mode: str,
    speed: float,
    output_path: Path,
) -> None:
    """
    Hybrid Dual-Engine (đúng tinh thần thiết kế gốc): thử GPU Fast Track trước
    (_render_video_only_gpu, NVENC qua ZeroGPU). Nếu hết quota ZeroGPU, ffmpeg không có
    NVENC, hay bất kỳ lỗi nào khác -> tự động CPU Graceful Fallback (libx264, luôn chạy
    được vì không phụ thuộc GPU).
    """
    try:
        _render_video_only_gpu(video_path, srt_path, style_label, aspect_mode, speed, output_path)
        if not (output_path.exists() and output_path.stat().st_size > 0):
            raise RuntimeError("File output GPU rỗng hoặc không được tạo ra.")
        logger.info("Render GPU (NVENC) thành công.")
    except Exception as exc:  # noqa: BLE001 - bất kỳ lỗi GPU nào cũng fallback, không làm sập pipeline
        logger.warning("Render GPU thất bại (%s) -> chuyển sang CPU fallback.", exc)
        _run_ffmpeg_render(video_path, srt_path, style_label, aspect_mode, speed, output_path, use_gpu=False)


# --------------------------------------------------------------------------- #
# CODE MỚI — Bước 4: Đồng bộ tốc độ audio (gốc / lồng tiếng) với speed của video
# --------------------------------------------------------------------------- #

def _extract_speed_corrected_audio(video_path: Path, output_path: Path, speed: float) -> None:
    """Trích audio gốc VÀ áp atempo=speed để khớp với video đã setpts theo speed."""
    run_subprocess(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-filter:a", f"atempo={speed:.4f}",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ],
        step_name="Trích + đồng bộ tốc độ audio gốc",
    )


def _speed_correct_audio(input_path: Path, output_path: Path, speed: float) -> None:
    """Áp atempo=speed cho 1 file audio bất kỳ (dùng cho track lồng tiếng đã ghép)."""
    run_subprocess(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-filter:a", f"atempo={speed:.4f}",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ],
        step_name="Đồng bộ tốc độ track lồng tiếng",
    )


# --------------------------------------------------------------------------- #
# CODE MỚI — Bước 5: Ghép track lồng tiếng KHỚP timestamp gốc (không nối tuần tự)
# --------------------------------------------------------------------------- #

def _build_synced_dub_track(
    segments: list[Segment],
    clip_paths: list[Path],
    total_duration_ms: int,
    min_gap_ms: int,
) -> AudioSegment:
    """
    Đặt từng clip audio TTS (đã sinh song song bởi _tts_generate_all_lines() THẬT) vào
    ĐÚNG mốc start_ms của segment gốc trong video, trên nền một track im lặng dài bằng
    video gốc. Khác hẳn generate_dubbed_audio_and_srt() thật vốn NỐI TUẦN TỰ các clip
    (bỏ qua timing gốc). `min_gap_ms` (từ slider "Khoảng nghỉ") chỉ dùng để tránh 2 câu
    liền kề đè hẳn lên nhau khi câu trước đọc lâu hơn khoảng cách tới câu sau — nếu vẫn
    còn chồng lấn nhẹ, 2 track sẽ được mix chồng (best-effort, không cắt mất lời thoại).
    """
    track = AudioSegment.silent(duration=max(total_duration_ms, 1))
    last_end_ms = 0
    for seg, clip_path in zip(segments, clip_paths):
        clip = AudioSegment.from_file(clip_path)
        desired_start_ms = int(seg.start * 1000)
        start_ms = desired_start_ms
        if last_end_ms and desired_start_ms < last_end_ms + min_gap_ms:
            start_ms = last_end_ms + min_gap_ms
        track = track.overlay(clip, position=start_ms)
        last_end_ms = start_ms + len(clip)
    return track


# --------------------------------------------------------------------------- #
# CODE MỚI — Caption & Hashtag (Gemini, best-effort)
# --------------------------------------------------------------------------- #

def _generate_social_caption(translated_text: str, short_title_mode: bool) -> str:
    """Dùng gemini_client THẬT (đã khởi tạo sẵn trong config.py) để tạo Caption &
    Hashtag. Lỗi (mạng, quota...) không được làm sập cả pipeline -> fallback."""
    if not translated_text.strip():
        return CAPTION_FALLBACK_TEXT

    length_instruction = (
        "Viết caption CỰC NGẮN GỌN, dưới 100 ký tự, kèm 3-5 hashtag viral."
        if short_title_mode
        else "Viết caption hấp dẫn (2-4 câu) kèm 5-8 hashtag viral phù hợp nội dung."
    )
    prompt = (
        "Bạn là chuyên gia content Shorts/TikTok tiếng Việt. Dựa trên nội dung video sau, "
        f"hãy viết 1 caption thu hút cho mạng xã hội.\n{length_instruction}\n\n"
        f"Nội dung video (đã dịch tiếng Việt):\n{translated_text[:2000]}"
    )
    try:
        response = gemini_client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)
        caption = (getattr(response, "text", "") or "").strip()
        return caption or CAPTION_FALLBACK_TEXT
    except Exception as exc:  # noqa: BLE001 - best-effort, không làm hỏng pipeline chính
        logger.warning("Tạo caption thất bại, dùng fallback: %s", exc)
        return CAPTION_FALLBACK_TEXT


# --------------------------------------------------------------------------- #
# Điều phối toàn bộ pipeline
# --------------------------------------------------------------------------- #

def process_video(
    video_file: str | None,
    douyin_text: str | None,
    subtitle_style_label: str,
    processing_mode: str,
    content_style: str,
    voice_label: str,
    speed: float,
    aspect_mode: str,
    tts_rate: float,
    tts_padding: float,
    keep_original_audio: bool,
    original_volume: float,
    short_title_mode: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple[dict, dict]:
    """
    Pipeline:
      xác định nguồn video (link/văn bản Douyin-TikTok ưu tiên, hoặc file upload)
      -> cache (tính TRƯỚC khi tải video nếu là link)
      -> (nếu link) tải video bằng yt-dlp -> validate
      -> tách âm thanh -> STT (Groq Whisper, CODE MỚI) -> dịch thuật theo lô (Gemini, hàm
         translate_script() THẬT của translate_and_tts.py)
      -> tạo Caption & Hashtag (Gemini, best-effort, CODE MỚI)
      -> dựng .srt theo ĐÚNG timestamp gốc (CODE MỚI)
      -> render video: canvas 3 lớp THẬT (build_canvas_filter()) + gắn cứng phụ đề bằng
         FFmpeg subtitles filter (CODE MỚI, tái sử dụng ASS_STYLE_PRESETS thật)
      -> (nếu bật Lồng tiếng) sinh audio TTS song song (_tts_generate_all_lines() THẬT)
         rồi đặt đúng vào timestamp gốc (CODE MỚI) -> đồng bộ tốc độ -> (tuỳ chọn) trộn
         với audio gốc bằng mix_audio() THẬT
      -> (nếu KHÔNG lồng tiếng) chỉ đồng bộ tốc độ audio gốc
      -> ghép video + audio bằng merge_video_audio() THẬT -> lưu cache -> trả kết quả
    """
    extracted_url = extract_clean_url(douyin_text)

    if extracted_url:
        source_mode = "url"
        uploaded_video_path: Path | None = None
    elif video_file:
        source_mode = "upload"
        uploaded_video_path = Path(video_file)
    else:
        raise gr.Error(
            "Vui lòng dán link/văn bản chia sẻ Douyin-TikTok HOẶC tải lên một file video "
            "(chỉ cần chọn MỘT trong hai cách)."
        )

    if douyin_text and douyin_text.strip() and not extracted_url:
        raise gr.Error(
            "Không tìm thấy link http(s) hợp lệ nào trong văn bản đã dán. "
            "Vui lòng kiểm tra lại nội dung copy từ Douyin/TikTok."
        )

    dubbing_enabled = processing_mode == MODE_SUBTITLE_AND_DUB
    voice_code = VOICE_OPTIONS.get(voice_label, VOICE_OPTIONS[DEFAULT_VOICE_LABEL])
    style_label = (
        subtitle_style_label if subtitle_style_label in ASS_STYLE_PRESETS else DEFAULT_SUBTITLE_STYLE_LABEL
    )
    aspect_mode = aspect_mode if aspect_mode in ASPECT_MODES else DEFAULT_ASPECT_MODE
    content_style = content_style if content_style in CONTENT_STYLES else DEFAULT_CONTENT_STYLE
    flip = aspect_mode != ASPECT_ORIGINAL

    progress(0.0, desc="Đang tính toán cache...")
    if source_mode == "url":
        file_hash = "douyinurl_" + hashlib.sha256(extracted_url.encode("utf-8")).hexdigest()
    else:
        file_hash = compute_sha256(uploaded_video_path)

    # build_cache_key() THẬT (config.py) chưa có tham số riêng cho content_style /
    # keep_original_audio / original_volume (tính năng MỚI). Nhúng thêm các giá trị này
    # vào 2 tham số subtitle_style_label / bgm_* chỉ để khoá cache không bị lẫn giữa các
    # tổ hợp tuỳ chọn khác nhau — KHÔNG ảnh hưởng tới style/BGM thật (BGM đã bỏ hoàn toàn).
    cache_style_key = f"{style_label}|cs={content_style}"
    cache_bgm_hash = f"keep_orig={keep_original_audio}" if dubbing_enabled else None
    cache_bgm_volume = original_volume if (dubbing_enabled and keep_original_audio) else 0.0

    cache_key = build_cache_key(
        file_hash, cache_style_key, dubbing_enabled, voice_code,
        speed, 0.0, flip, 0.0, 0.0,
        0.0, cache_bgm_hash, cache_bgm_volume,
        aspect_mode, tts_rate, tts_padding,
        source_url=extracted_url,
    )

    cached_single = get_cached_output(cache_key)
    if cached_single is not None:
        cached_caption = get_cached_caption(cache_key) or CAPTION_FALLBACK_TEXT
        progress(1.0, desc="Đã tìm thấy kết quả trong cache!")
        save_to_history(cached_single)
        return (
            gr.update(value=str(cached_single), visible=True),
            gr.update(value=cached_caption),
        )

    with tempfile.TemporaryDirectory(prefix="cn2vi_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        try:
            if source_mode == "url":
                progress(0.03, desc="Đang tải video từ link Douyin/TikTok (yt-dlp)...")
                video_path = download_video_from_url(extracted_url, tmp_dir)
            else:
                video_path = uploaded_video_path

            progress(0.06, desc="Đang kiểm tra file đầu vào...")
            video_duration_sec = validate_video(video_path)
            total_duration_ms = int(video_duration_sec * 1000)

            progress(0.10, desc="Đang tách âm thanh để nhận diện giọng nói...")
            stt_audio_path = tmp_dir / "audio_stt.wav"
            _extract_audio_for_stt(video_path, stt_audio_path)

            progress(0.22, desc="Đang nhận diện giọng nói (Groq Whisper)...")
            segments = _transcribe_audio_groq(stt_audio_path)

            progress(0.35, desc="Đang dịch thuật sang tiếng Việt (Gemini)...")
            texts_zh = [seg.text_zh for seg in segments]
            texts_vi = translate_script(texts_zh, content_style=content_style)
            for seg, vi in zip(segments, texts_vi):
                seg.text_vi = vi

            progress(0.42, desc="Đang tạo mẫu Caption & Hashtag (Gemini)...")
            combined_translated_text = " ".join(t for t in texts_vi if t.strip())
            caption_text = _generate_social_caption(combined_translated_text, short_title_mode)
            store_caption_in_cache(cache_key, caption_text)

            progress(0.48, desc="Đang tạo file phụ đề .srt...")
            srt_path = tmp_dir / "subtitles.srt"
            srt_path.write_text(_build_srt_from_segments(segments), encoding="utf-8")

            progress(0.55, desc="Đang render video (hiệu ứng + phụ đề cứng)...")
            video_only_path = tmp_dir / "video_only.mp4"
            _render_video_only(video_path, srt_path, style_label, aspect_mode, speed, video_only_path)

            if dubbing_enabled:
                progress(0.68, desc="Đang tạo giọng lồng tiếng AI (edge-tts, song song)...")
                tts_rate_str = f"{int(tts_rate):+d}%"
                clip_paths = _run_async(
                    _tts_generate_all_lines(texts_vi, voice_code, tts_rate_str, tmp_dir)
                )

                progress(0.80, desc="Đang ghép track lồng tiếng khớp timeline video gốc...")
                min_gap_ms = int(tts_padding * 1000)
                dub_track = _build_synced_dub_track(segments, clip_paths, total_duration_ms, min_gap_ms)
                dub_track_raw_path = tmp_dir / "dub_track_raw.wav"
                dub_track.export(dub_track_raw_path, format="wav")

                dub_track_sped_path = tmp_dir / "dub_track_sped.aac"
                _speed_correct_audio(dub_track_raw_path, dub_track_sped_path, speed)

                if keep_original_audio:
                    progress(0.88, desc="Đang trộn giọng lồng tiếng với âm thanh gốc...")
                    original_audio_sped_path = tmp_dir / "original_audio_sped.aac"
                    _extract_speed_corrected_audio(video_path, original_audio_sped_path, speed)
                    mixed_audio_path = tmp_dir / "mixed_audio.aac"
                    _run_async(
                        mix_audio(
                            tts_audio_path=str(dub_track_sped_path),
                            original_video_path=str(original_audio_sped_path),
                            output_audio_path=str(mixed_audio_path),
                            keep_original_audio=True,
                            original_volume=original_volume,
                        )
                    )
                else:
                    mixed_audio_path = dub_track_sped_path
            else:
                progress(0.75, desc="Đang trích + đồng bộ tốc độ audio gốc...")
                mixed_audio_path = tmp_dir / "original_audio_sped.aac"
                _extract_speed_corrected_audio(video_path, mixed_audio_path, speed)

            progress(0.94, desc="Đang ghép video hoàn chỉnh...")
            final_path = tmp_dir / "final_output.mp4"
            _run_async(merge_video_audio(str(video_only_path), str(mixed_audio_path), str(final_path)))

            progress(0.98, desc="Đang lưu vào cache...")
            stored_path = store_in_cache(cache_key, final_path)
            save_to_history(stored_path)

            progress(1.0, desc="Hoàn tất!")
            return (
                gr.update(value=str(stored_path), visible=True),
                gr.update(value=caption_text),
            )

        except (PipelineError, TTSPipelineError) as exc:
            logger.error("Pipeline thất bại: %s", exc)
            raise gr.Error(str(exc)) from exc
        except gr.Error:
            raise
        except Exception as exc:  # noqa: BLE001 - chuyển mọi lỗi không lường trước thành thông báo cho người dùng
            logger.exception("Lỗi không xác định trong pipeline")
            raise gr.Error(f"Đã xảy ra lỗi không mong muốn: {exc}") from exc


# --------------------------------------------------------------------------- #
# CODE MỚI — Endpoint API riêng "process_reup" cho frontend tĩnh (GitHub Pages, index.html
# gọi qua @gradio/client). Frontend đó CHỈ gửi 3 tham số tối giản (share_text,
# keep_original_audio, original_audio_volume) — mọi tuỳ chọn khác dùng giá trị mặc định
# của UI chính. Đây chỉ là lớp bọc mỏng gọi lại process_video(), KHÔNG lặp logic.
# --------------------------------------------------------------------------- #

def process_reup(
    share_text: str,
    keep_original_audio: bool,
    original_audio_volume: float,
    progress: gr.Progress = gr.Progress(),
) -> tuple[dict, dict]:
    """
    api_name="process_reup" — khớp đúng API_NAME="/process_reup" mà index.html (frontend
    tĩnh) đang gọi, đúng thứ tự tham số [shareText, keepOriginal, volume] mà file đó gửi
    qua client.submit(). Luôn xử lý theo link/văn bản chia sẻ (không nhận upload file trực
    tiếp từ frontend này), chế độ "Cả Phụ đề + Lồng tiếng" (vì frontend chỉ có control cho
    giữ/chỉnh âm lượng audio gốc — vốn chỉ có ý nghĩa khi có lồng tiếng), các tuỳ chọn còn
    lại (kiểu khung phụ đề, văn phong dịch, giọng đọc, speed, khung hình, tts rate/padding,
    chế độ tiêu đề ngắn) dùng đúng giá trị mặc định của UI chính.
    """
    return process_video(
        video_file=None,
        douyin_text=share_text,
        subtitle_style_label=DEFAULT_SUBTITLE_STYLE_LABEL,
        processing_mode=MODE_SUBTITLE_AND_DUB,
        content_style=DEFAULT_CONTENT_STYLE,
        voice_label=DEFAULT_VOICE_LABEL,
        speed=SPEED_DEFAULT,
        aspect_mode=DEFAULT_ASPECT_MODE,
        tts_rate=TTS_RATE_DEFAULT,
        tts_padding=TTS_PADDING_DEFAULT,
        keep_original_audio=keep_original_audio,
        original_volume=original_audio_volume,
        short_title_mode=False,
        progress=progress,
    )


# --------------------------------------------------------------------------- #
# Giao diện Gradio
# --------------------------------------------------------------------------- #

# Gradio 6.0: theme/css KHÔNG còn được truyền vào gr.Blocks() nữa (gây UserWarning),
# mà phải truyền vào demo.launch() ở cuối file. Ở đây chỉ định nghĩa nội dung để dùng sau.
APP_THEME = gr.themes.Soft()
APP_CSS = """
#title { text-align: center; margin-bottom: 0.5rem; }
#subtitle { text-align: center; color: #6b7280; margin-bottom: 1.5rem; }

/* --------------------------------------------------------------------- */
/* FIX LỖI CUỘN/LƯỚT TRÊN DI ĐỘNG (MOBILE SCROLL FIX) */
/* --------------------------------------------------------------------- */
html, body {
    height: auto !important;
    min-height: 100% !important;
    overflow-y: auto !important;
    -webkit-overflow-scrolling: touch !important;
    touch-action: pan-y !important;
}

gradio-app, #root, .gradio-container, .app, .main, .wrap {
    height: auto !important;
    min-height: 100% !important;
    max-height: none !important;
    overflow-y: auto !important;
    touch-action: pan-y !important;
}

div[data-testid="block"], .block, .form, .tabitem {
    max-height: none !important;
    overflow: visible !important;
}

@media (max-width: 1024px) {
    html, body, gradio-app, .gradio-container, #root, .app, .main, .wrap {
        position: static !important;
        height: auto !important;
        min-height: 100% !important;
        max-height: none !important;
        overflow: visible !important;
        overflow-y: auto !important;
        touch-action: pan-y !important;
        -webkit-overflow-scrolling: touch !important;
    }

    div[data-testid="block"], .block, .form, .tabitem {
        max-height: none !important;
        overflow: visible !important;
    }
}
"""


def refresh_history_ui() -> dict:
    """Callback cho nút '🔄 Tải lại danh sách cache' và khi trang load: nạp lại danh sách."""
    names = list_cached_output_names()
    return gr.update(choices=names, value=(names[0] if names else None))


def preview_history_video(filename: str | None) -> str | None:
    """Callback khi người dùng chọn 1 video trong dropdown lịch sử -> hiện lên player."""
    if not filename:
        return None
    path = CACHED_OUTPUTS_DIR / filename
    return str(path) if path.exists() else None


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Dịch & Lồng tiếng Video Trung-Việt") as demo:
        gr.Markdown("# 🎬 Dịch, Gắn Phụ Đề & Lồng Tiếng Video: Trung → Việt", elem_id="title")
        gr.Markdown(
            f"Tải lên video tiếng Trung (tối đa {MAX_FILE_SIZE_MB}MB, "
            f"{MAX_DURATION_SEC // 60} phút) để tự động dịch phụ đề, gắn khung che sub gốc "
            f"và lồng tiếng AI tiếng Việt.",
            elem_id="subtitle",
        )

        with gr.Row():
            with gr.Column(scale=1):
                douyin_text_input = gr.Textbox(
                    label="Dán Link hoặc Văn bản chia sẻ Douyin / TikTok",
                    placeholder=(
                        "Ví dụ: 5.69 复制打开抖音，看看【落雨🌂的作品】... "
                        "https://v.douyin.com/wUNGqz3pu_w/ 08/23 :2pm — dán nguyên đoạn copy cũng được, "
                        "hệ thống tự lọc lấy link."
                    ),
                    lines=2,
                )
                gr.Markdown(
                    "⬆️ **Chọn 1 trong 2**: dán link/văn bản Douyin-TikTok ở trên, "
                    "HOẶC tải file video lên ở dưới. Nếu có dán văn bản, hệ thống sẽ ưu tiên "
                    "trích xuất link và tải video từ link trước, bỏ qua file tải lên (nếu có)."
                )
                video_input = gr.Video(label="Video đầu vào (tiếng Trung) — Upload file")

                subtitle_style_radio = gr.Radio(
                    choices=list(ASS_STYLE_PRESETS.keys()),
                    value=DEFAULT_SUBTITLE_STYLE_LABEL,
                    label="Kiểu khung phụ đề (che phụ đề gốc tiếng Trung)",
                )

                aspect_mode_radio = gr.Radio(
                    choices=ASPECT_MODES,
                    label="Chế độ khung hình CapCut Canvas 9:16 (Shorts/Reels/TikTok)",
                    value=DEFAULT_ASPECT_MODE,
                    info=(
                        "Cả 2 chế độ Canvas 9:16 đều dùng chung 1 pipeline (nền mờ lấp đầy "
                        "khung + video chính giữ nguyên ở giữa, lật ngang tự động)."
                    ),
                )

                content_style_radio = gr.Radio(
                    choices=list(CONTENT_STYLES.keys()),
                    value=DEFAULT_CONTENT_STYLE,
                    label="Văn phong dịch (drama / comedy / kids)",
                    info="Quyết định giọng văn khi Gemini dịch câu thoại sang tiếng Việt.",
                )

                short_title_mode = gr.Checkbox(
                    label="⚡ Chế độ Tiêu đề Ngắn (Ép tối đa ~100 ký tự)",
                    value=False,
                    info="Bật để ép Caption/Tiêu đề tự động tạo (Gemini) ngắn gọn hơn.",
                )

                mode_radio = gr.Radio(
                    choices=PROCESSING_MODES,
                    value=MODE_SUBTITLE_ONLY,
                    label="Chế độ xử lý",
                )

                voice_radio = gr.Radio(
                    choices=list(VOICE_OPTIONS.keys()),
                    value=DEFAULT_VOICE_LABEL,
                    label="Giọng lồng tiếng AI (tiếng Việt)",
                    visible=False,
                )

                tts_rate_slider = gr.Slider(
                    minimum=TTS_RATE_MIN, maximum=TTS_RATE_MAX, value=TTS_RATE_DEFAULT, step=1,
                    label="Tốc độ đọc giọng AI (Rate %)",
                    visible=False,
                )
                tts_padding_slider = gr.Slider(
                    minimum=TTS_PADDING_MIN, maximum=TTS_PADDING_MAX, value=TTS_PADDING_DEFAULT, step=0.05,
                    label="Khoảng nghỉ tối thiểu giữa các câu thoại (giây)",
                    visible=False,
                )
                keep_original_audio_checkbox = gr.Checkbox(
                    value=False,
                    label="Giữ thêm âm thanh gốc (trộn cùng lồng tiếng)",
                    visible=False,
                )
                original_volume_slider = gr.Slider(
                    minimum=ORIGINAL_VOLUME_MIN, maximum=ORIGINAL_VOLUME_MAX,
                    value=ORIGINAL_VOLUME_DEFAULT, step=0.01,
                    label="Âm lượng âm thanh gốc (khi trộn cùng lồng tiếng)",
                    visible=False,
                )

                # Chỉ hiện chọn giọng đọc + rate/padding/giữ audio gốc khi bật chế độ Lồng tiếng.
                mode_radio.change(
                    fn=lambda mode: (
                        gr.update(visible=(mode == MODE_SUBTITLE_AND_DUB)),
                        gr.update(visible=(mode == MODE_SUBTITLE_AND_DUB)),
                        gr.update(visible=(mode == MODE_SUBTITLE_AND_DUB)),
                        gr.update(visible=(mode == MODE_SUBTITLE_AND_DUB)),
                        gr.update(visible=(mode == MODE_SUBTITLE_AND_DUB)),
                    ),
                    inputs=[mode_radio],
                    outputs=[
                        voice_radio, tts_rate_slider, tts_padding_slider,
                        keep_original_audio_checkbox, original_volume_slider,
                    ],
                )

                with gr.Accordion("🎨 Tùy chỉnh tốc độ phát", open=False):
                    speed_slider = gr.Slider(
                        minimum=SPEED_MIN, maximum=SPEED_MAX, value=SPEED_DEFAULT, step=0.01,
                        label="Tốc độ phát (Speed)",
                    )

                submit_btn = gr.Button("🚀 Bắt đầu xử lý", variant="primary")

                gr.Markdown(
                    "**Quy trình:** Kiểm tra file → Tách âm thanh → "
                    "Nhận diện giọng nói (Groq Whisper) → Dịch thuật (Gemini) → "
                    "Tạo phụ đề .srt (khớp timestamp gốc) → Render video (canvas + phụ đề "
                    "cứng) → Track thoại (gốc / lồng tiếng AI khớp timestamp) → Ghép hoàn chỉnh."
                )

            with gr.Column(scale=1):
                video_output = gr.Video(label="🎬 Kết quả")

                caption_output = gr.Code(
                    label="📌 Mẫu Caption & Hashtag Đăng Video (Tự động tạo)",
                    language=None,
                    interactive=False,
                )

        submit_btn.click(
            fn=process_video,
            inputs=[
                video_input, douyin_text_input, subtitle_style_radio, mode_radio,
                content_style_radio, voice_radio, speed_slider, aspect_mode_radio,
                tts_rate_slider, tts_padding_slider,
                keep_original_audio_checkbox, original_volume_slider, short_title_mode,
            ],
            outputs=[video_output, caption_output],
        )

        # --- CODE MỚI: endpoint API riêng cho frontend tĩnh index.html (GitHub Pages) ---
        # 3 component ẩn (visible=False) + 1 nút ẩn chỉ để expose api_name="process_reup"
        # cho @gradio/client gọi trực tiếp — KHÔNG hiện trên giao diện chính, không ảnh
        # hưởng tới form đầy đủ ở trên. Thứ tự input PHẢI khớp đúng thứ tự index.html gửi:
        # client.submit("/process_reup", [shareText, keepOriginal, volume]).
        api_share_text = gr.Textbox(visible=False)
        api_keep_original_audio = gr.Checkbox(visible=False)
        api_original_volume = gr.Slider(
            minimum=ORIGINAL_VOLUME_MIN, maximum=ORIGINAL_VOLUME_MAX,
            value=ORIGINAL_VOLUME_DEFAULT, visible=False,
        )
        api_trigger_btn = gr.Button(visible=False)
        api_trigger_btn.click(
            fn=process_reup,
            inputs=[api_share_text, api_keep_original_audio, api_original_volume],
            outputs=[video_output, caption_output],
            api_name="process_reup",
        )

        with gr.Accordion("📂 Lịch sử Video đã render (Lưu 24h)", open=False):
            gr.Markdown(
                "Mọi video render **thành công** được tự động lưu tạm ở đây trong "
                "**24 giờ**. Lỡ thoát hoặc F5 trình duyệt thì mở lại trang, bấm "
                "'🔄 Tải lại danh sách cache' là thấy ngay video cũ và tải về luôn, "
                "không cần chờ render lại từ đầu. File quá 24h sẽ tự động bị xóa."
            )
            with gr.Row():
                history_dropdown = gr.Dropdown(
                    label="Chọn video trong lịch sử",
                    choices=[],
                    value=None,
                    interactive=True,
                    scale=3,
                )
                refresh_history_btn = gr.Button("🔄 Tải lại danh sách cache", scale=1)

            history_video_player = gr.Video(
                label="Video lịch sử (bấm nút tải ⬇️ ở góc player để tải về máy)"
            )

        refresh_history_btn.click(fn=refresh_history_ui, outputs=[history_dropdown])
        history_dropdown.change(
            fn=preview_history_video,
            inputs=[history_dropdown],
            outputs=[history_video_player],
        )
        # Tự động nạp danh sách lịch sử ngay khi người dùng mở trang (F5/quay lại).
        demo.load(fn=refresh_history_ui, outputs=[history_dropdown])

    return demo


if __name__ == "__main__":
    interface = build_interface()
    interface.queue(max_size=5, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
        theme=APP_THEME,
        css=APP_CSS,
    )
