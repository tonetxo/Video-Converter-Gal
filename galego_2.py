# --- Video Converter Pro+ (Versión Final, Completa e Funcional) ---

import gradio as gr
import subprocess
import os
import sys
import shutil
import tempfile
import json
import multiprocessing
from datetime import datetime
from PIL import Image
import concurrent.futures
import math
import hashlib
import time
from pathlib import Path
import logging

# --- Configuration and Constants ---
FFMPEG_PATH = shutil.which("ffmpeg")
FFPROBE_PATH = shutil.which("ffprobe")
CPU_CORES = multiprocessing.cpu_count()
MAX_THREADS = min(CPU_CORES, 16)
CACHE_DIR = Path.home() / ".video_converter_cache"
TEMP_DIR = Path(tempfile.gettempdir()) / "video_converter_temp"
PRESETS_FILE = Path.home() / ".video_converter_presets.json"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Color correction presets
COLOR_PRESETS = {
    "Normal": {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0, "sharp": 0.0, "blur": 0.0, "gamma": 1.0},
    "Vibrant": {"brightness": 0.02, "contrast": 1.15, "saturation": 1.25, "sharp": 0.5, "blur": 0.0, "gamma": 1.1},
    "Soft": {"brightness": 0.05, "contrast": 0.95, "saturation": 0.9, "sharp": 0.0, "blur": 0.8, "gamma": 1.0},
    "High Contrast": {"brightness": 0.0, "contrast": 1.25, "saturation": 1.1, "sharp": 1.0, "blur": 0.0, "gamma": 1.0},
    "Desaturated": {"brightness": 0.0, "contrast": 1.05, "saturation": 0.7, "sharp": 0.0, "blur": 0.0, "gamma": 1.0},
    "Warm": {"brightness": 0.03, "contrast": 1.05, "saturation": 1.15, "sharp": 0.3, "blur": 0.0, "gamma": 1.05},
    "Cool": {"brightness": -0.02, "contrast": 1.1, "saturation": 1.05, "sharp": 0.4, "blur": 0.0, "gamma": 0.95}
}

# --- Preset Management Functions ---
def load_presets_from_file():
    if PRESETS_FILE.exists():
        with open(PRESETS_FILE, 'r', encoding='utf-8') as f:
            try: return json.load(f)
            except json.JSONDecodeError: return {}
    return {}

def save_presets_to_file(presets):
    with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
        json.dump(presets, f, indent=4)

def get_preset_choices():
    return list(load_presets_from_file().keys())

def save_preset(preset_name, *values):
    if not preset_name:
        gr.Warning("Por favor, introduce un nome para o axuste.")
        return gr.update(choices=get_preset_choices())
    presets = load_presets_from_file()
    keys = [
        "speed", "quality", "gpu", "scale_enabled", "scale_factor", "fixed_res_enabled", "fixed_res_value",
        "interpolate", "custom_fps", "fps_value", "brightness", "contrast", "saturation", "sharp",
        "blur", "gamma", "process_audio", "preserve_pitch", "fade_in_1", "fade_out_1",
        "fade_in_2", "fade_out_2", "vol_original", "vol_track_1", "vol_track_2"
    ]
    presets[preset_name] = dict(zip(keys, values))
    save_presets_to_file(presets)
    gr.Info(f"Axuste '{preset_name}' gardado con éxito!")
    return gr.update(choices=get_preset_choices(), value=preset_name)

def load_preset(preset_name):
    presets = load_presets_from_file()
    if preset_name in presets:
        gr.Info(f"Cargando axuste '{preset_name}'...")
        p = presets[preset_name]
        return [
            p.get("speed", 1.0), p.get("quality", 23), p.get("gpu", True),
            p.get("scale_enabled", False), p.get("scale_factor", 100),
            p.get("fixed_res_enabled", False), p.get("fixed_res_value", ""),
            p.get("interpolate", False), p.get("custom_fps", False),
            p.get("fps_value", 25), p.get("brightness", 0.0),
            p.get("contrast", 1.0), p.get("saturation", 1.0),
            p.get("sharp", 0.0), p.get("blur", 0.0), p.get("gamma", 1.0),
            p.get("process_audio", True), p.get("preserve_pitch", True),
            p.get("fade_in_1", 1.0), p.get("fade_out_1", 1.0),
            p.get("fade_in_2", 1.0), p.get("fade_out_2", 1.0),
            p.get("vol_original", 1.0), p.get("vol_track_1", 1.0),
            p.get("vol_track_2", 1.0)
        ]
    gr.Warning(f"Non se atopou o axuste '{preset_name}'.")
    return [None] * 25

def delete_preset(preset_name):
    if not preset_name:
        gr.Warning("Por favor, elixe un axuste para eliminar.")
        return gr.update(choices=get_preset_choices())
    presets = load_presets_from_file()
    if preset_name in presets:
        del presets[preset_name]
        save_presets_to_file(presets)
        gr.Info(f"Axuste '{preset_name}' eliminado.")
        return gr.update(choices=get_preset_choices(), value=None)
    gr.Warning(f"Non se puido atopar o axuste '{preset_name}' para eliminar.")
    return gr.update(choices=get_preset_choices())

# --- Core Processing Functions ---
def verify_tools():
    if not all([FFMPEG_PATH, FFPROBE_PATH]):
        raise gr.Error("FFmpeg e FFprobe son necesarios.")
    CACHE_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    logger.info("Ferramentas verificadas correctamente")

def get_video_duration(file_path):
    try:
        cmd = [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        return float(result.stdout.strip())
    except: return 0.0

def get_audio_duration(file_path):
    try:
        cmd = [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        return float(result.stdout.strip())
    except: return 0.0

def get_video_info(file_path):
    if not file_path or not Path(file_path).exists(): return "O ficheiro non existe."
    try:
        cmd = [FFPROBE_PATH, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(file_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        data = json.loads(result.stdout)
        video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
        if video_stream:
            info = [f"**Formato:** {data['format'].get('format_long_name', 'N/A')}"]
            fps_str = video_stream.get('r_frame_rate', '0/1')
            try: num, den = map(int, fps_str.split('/')); fps = f"{num / den:.2f}" if den > 0 else "0"
            except: fps = "N/A"
            info.extend([f"**Resolución:** {video_stream.get('width', 'N/A')}x{video_stream.get('height', 'N/A')}", f"**FPS:** {fps}", f"**Códec de vídeo:** {video_stream.get('codec_name', 'N/A')}"])
            frames = video_stream.get('nb_frames', 'N/A')
            if frames != 'N/A': info.append(f"**Número de fotogramas:** {frames}")
            audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
            if audio_stream:
                info.append(f"**Códec de audio:** {audio_stream.get('codec_name', 'N/A')}")
                sample_rate = audio_stream.get('sample_rate', 'N/A')
                if sample_rate != 'N/A': info.append(f"**Frecuencia de mostraxe:** {sample_rate} Hz")
            else: info.append("**Audio:** Sen pista de audio")
            duration = float(data['format'].get('duration', 0))
            info.append(f"**Duración:** {duration:.2f} segundos")
            bitrate = data['format'].get('bit_rate', 'N/A')
            if bitrate != 'N/A': info.append(f"**Taxa de bits:** {int(bitrate) / 1_000_000:.2f} Mbps")
            return "\n".join(info)
    except Exception: pass
    try:
        with Image.open(file_path) as img:
            w, h = img.size
            n_frames = getattr(img, 'n_frames', 1)
            duration_ms = img.info.get('duration', 40)
            total_s = (n_frames * duration_ms) / 1000.0 if n_frames > 1 else 0
            fps = n_frames / total_s if total_s > 0 else 0
            return "\n".join([f"**Formato:** Imaxe animada ({img.format})", f"**Resolución:** {w}x{h}", f"**FPS:** {fps:.2f}", f"**Número de fotogramas:** {n_frames}"])
    except Exception as e: return f"Non se puido ler o ficheiro.\nErro: {e}"

def detect_gpu():
    try:
        result = subprocess.run([FFMPEG_PATH, "-encoders"], capture_output=True, text=True, check=True, encoding='utf-8')
        return "h264_nvenc" in result.stdout
    except: return False

def verify_audio_filters():
    try:
        result = subprocess.run([FFMPEG_PATH, "-filters"], capture_output=True, text=True, check=True, encoding='utf-8')
        return "atempo" in result.stdout, "rubberband" in result.stdout
    except: return False, False

def get_original_fps(file_path):
    try:
        cmd_fps = [FFPROBE_PATH, "-v", "error", "-select_streams", "v:0", "-of", "default=noprint_wrappers=1:nokey=1", "-show_entries", "stream=r_frame_rate", str(file_path)]
        fps_str = subprocess.run(cmd_fps, check=True, capture_output=True, text=True, encoding='utf-8').stdout.strip()
        num, den = map(int, fps_str.split('/'))
        return num / den if den > 0 else 25.0
    except:
        try:
            with Image.open(file_path) as img:
                n_frames = getattr(img, 'n_frames', 1)
                duration_info = img.info.get('duration', 40)
                if n_frames <= 1 or duration_info <= 0: return 25.0
                return n_frames / (n_frames * duration_info / 1000.0)
        except: return 25.0

def scale_frame(args):
    input_path, output_path, scale_factor = args
    try:
        with Image.open(input_path) as img:
            new_width = int(img.width * scale_factor / 100)
            new_height = int(img.height * scale_factor / 100)
            img.resize((new_width, new_height), Image.Resampling.LANCZOS).save(output_path, optimize=True)
    except Exception as e: logger.error(f"Erro ao escalar {input_path}: {e}")

def generate_crop_filter_for_aspect_ratio(w, h):
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"

def validate_color_parameters(b, c, s, sharp, blur, gamma):
    return (max(-0.3, min(0.3, b)), max(0.7, min(1.5, c)), max(0.5, min(2.0, s)), max(0.0, min(10.0, sharp)), max(0.0, min(10.0, blur)), max(0.1, min(10.0, gamma)))

def preview_color_correction(input_tempfile, b, c, s, sharp, blur, gamma):
    if not input_tempfile: raise gr.Error("Por favor, sube un ficheiro primeiro.")
    input_path = input_tempfile.name
    b, c, s, sharp, blur, gamma = validate_color_parameters(b, c, s, sharp, blur, gamma)
    original_frame, corrected_frame = TEMP_DIR / f"original_{time.time()}.png", TEMP_DIR / f"corrected_{time.time()}.png"
    try:
        with Image.open(input_path) as img: img.seek(0); img.copy().convert("RGB").save(original_frame)
    except:
        subprocess.run([FFMPEG_PATH, "-i", input_path, "-vframes", "1", "-q:v", "2", "-y", str(original_frame)], check=True, capture_output=True)
    filter_chain, eq_params = [], []
    if b != 0.0: eq_params.append(f"brightness={b:.3f}")
    if c != 1.0: eq_params.append(f"contrast={c:.3f}")
    if s != 1.0: eq_params.append(f"saturation={s:.3f}")
    if gamma != 1.0: eq_params.append(f"gamma={gamma:.3f}")
    if eq_params: filter_chain.append("eq=" + ":".join(eq_params))
    if blur > 0.0: filter_chain.append(f"boxblur=luma_radius={blur:.3f}:luma_power=1")
    if sharp > 0.0: filter_chain.append(f"unsharp=lx=5:ly=5:la={sharp:.3f}")
    if not filter_chain: shutil.copy(original_frame, corrected_frame)
    else:
        try:
            subprocess.run([FFMPEG_PATH, "-i", str(original_frame), "-vf", ",".join(filter_chain), "-y", str(corrected_frame)], check=True, capture_output=True)
        except subprocess.CalledProcessError as e: raise gr.Error(f"A corrección de cor fallou: {e.stderr.decode()}")
    return str(original_frame), str(corrected_frame)

def apply_color_preset(name):
    p = COLOR_PRESETS.get(name, COLOR_PRESETS["Normal"])
    return p["brightness"], p["contrast"], p["saturation"], p["sharp"], p["blur"], p["gamma"]

def process_audio_track(track_id, temp_file, fade_in, fade_out, target_duration, speed, preserve_pitch):
    if not temp_file: return None
    track_path, mtime = temp_file.name, Path(temp_file.name).stat().st_mtime
    key = hashlib.sha256(f"{track_path}-{mtime}-{fade_in}-{fade_out}-{target_duration}-{speed}-{preserve_pitch}".encode()).hexdigest()
    cached_path = CACHE_DIR / f"audio_{track_id}_{key}.aac"
    if cached_path.exists(): return str(cached_path)
    orig_dur = get_audio_duration(track_path)
    req_speed = orig_dur / target_duration if orig_dur > 0 else 1.0
    filters, current_speed = [], req_speed
    if abs(req_speed - 1.0) > 0.01:
        if preserve_pitch and verify_audio_filters()[1]: filters.append(f"rubberband=tempo={req_speed:.4f}")
        elif preserve_pitch and verify_audio_filters()[0]:
            while current_speed > 2.0: filters.append("atempo=2.0"); current_speed /= 2.0
            while current_speed < 0.5: filters.append("atempo=0.5"); current_speed /= 0.5
            if abs(current_speed - 1.0) > 0.01: filters.append(f"atempo={current_speed:.4f}")
        else: filters.append(f"atempo={req_speed:.4f}")
    fade_out_start = max(0, target_duration - fade_out)
    filters.extend([f"afade=t=in:st=0:d={fade_in:.2f}", f"afade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f}"])
    cmd = [FFMPEG_PATH, "-i", track_path] + (["-af", ",".join(filters)] if filters else []) + ["-t", str(target_duration), "-c:a", "aac", "-b:a", "192k", "-y", str(cached_path)]
    try: subprocess.run(cmd, check=True, capture_output=True); return str(cached_path)
    except subprocess.CalledProcessError as e: logger.error(f"Procesamento de audio {track_id} fallou: {e}"); return None

def process_video(input_tempfile, output_path, speed=1.0, quality_crf=23, use_gpu=False,
                  scale_enabled=False, scale_factor=100, use_custom_fps=False, fps_value=25.0,
                  interpolate=False, use_fixed_res=False, fixed_resolution="", process_audio=False,
                  preserve_pitch=False,
                  audio_track_1_tempfile=None, fade_in_1=1.0, fade_out_1=1.0,
                  audio_track_2_tempfile=None, fade_in_2=1.0, fade_out_2=1.0,
                  vol_original=1.0, vol_track_1=1.0, vol_track_2=1.0,
                  brightness=0.0, contrast=1.0, saturation=1.0,
                  sharp=0.0, blur=0.0, gamma=1.0,
                  progress=gr.Progress(track_tqdm=True)):
    verify_tools()
    if not input_tempfile: raise gr.Error("Por favor, sube un ficheiro.")
    input_path = Path(input_tempfile.name)
    brightness, contrast, saturation, sharp, blur, gamma = validate_color_parameters(brightness, contrast, saturation, sharp, blur, gamma)
    input_info = get_video_info(input_path)
    is_conventional_video = input_path.suffix.lower() in [".mp4",".mov",".avi",".mkv",".webm"]
    original_fps, original_duration = get_original_fps(input_path), get_video_duration(input_path)
    if original_duration == 0:
        try:
            with Image.open(input_path) as img:
                n_frames = getattr(img, 'n_frames', 1)
                duration_ms = img.info.get('duration', 40)
                original_duration = (n_frames * duration_ms) / 1000.0 if n_frames > 1 else 5.0
        except: original_duration = 5.0
    logger.info(f"Procesando: {input_path} (Duración: {original_duration:.2f}s, FPS: {original_fps:.2f})")
    mtime = input_path.stat().st_mtime
    cache_key = hashlib.sha256(f"{input_path}-{mtime}-{scale_enabled}-{scale_factor}".encode()).hexdigest()
    cache_path, frame_source = CACHE_DIR / cache_key, (CACHE_DIR / cache_key) / "frame-%05d.png"
    if not cache_path.exists():
        cache_path.mkdir(parents=True)
        original_frames_dir, original_frame_pattern = cache_path / "original_frames", (cache_path / "original_frames") / "frame-%05d.png"
        original_frames_dir.mkdir()
        if is_conventional_video: subprocess.run([FFMPEG_PATH, "-i", str(input_path), "-q:v", "2", str(original_frame_pattern)], check=True, capture_output=True)
        else:
            with Image.open(input_path) as img:
                for i in progress.tqdm(range(getattr(img, 'n_frames', 1)), desc="Extraendo fotogramas"):
                    img.seek(i); img.copy().convert("RGB").save(original_frames_dir / f"frame-{i:05d}.png")
        if scale_enabled and scale_factor != 100:
            frame_files = sorted(original_frames_dir.glob("frame-*.png"))
            args_list = [(str(f), str(cache_path / f.name), scale_factor) for f in frame_files]
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                list(progress.tqdm(executor.map(scale_frame, args_list), total=len(args_list), desc="Escalando"))
            shutil.rmtree(original_frames_dir)
        else:
            for f in original_frames_dir.glob("frame-*.png"): shutil.move(str(f), cache_path / f.name)
            original_frames_dir.rmdir()
    
    vf_filters, final_video_duration, input_ffmpeg_rate = [], original_duration / speed, original_fps
    
    if speed != 1.0:
        vf_filters.append(f"setpts={1.0/speed:.4f}*PTS")
        input_ffmpeg_rate = original_fps * speed
    
    # FIX: A lóxica de filtros de vídeo corrixida
    if speed < 1.0 and interpolate:
        vf_filters.append(f"minterpolate=fps={float(fps_value):.2f}")
    elif use_custom_fps and abs(float(fps_value) - (input_ffmpeg_rate if speed != 1.0 else original_fps)) > 0.01:
        vf_filters.append(f"fps={float(fps_value):.2f}")

    eq_params = []
    if brightness != 0.0: eq_params.append(f"brightness={brightness:.3f}")
    if contrast != 1.0: eq_params.append(f"contrast={contrast:.3f}")
    if saturation != 1.0: eq_params.append(f"saturation={saturation:.3f}")
    if gamma != 1.0: eq_params.append(f"gamma={gamma:.3f}")
    if eq_params: vf_filters.append("eq=" + ":".join(eq_params))
    if blur > 0.0: vf_filters.append(f"boxblur=luma_radius={blur:.3f}:luma_power=1")
    if sharp > 0.0: vf_filters.append(f"unsharp=lx=5:ly=5:la={sharp:.3f}")

    if use_fixed_res and 'x' in fixed_resolution:
        try: w, h = map(int, fixed_resolution.lower().split('x')); vf_filters.append(generate_crop_filter_for_aspect_ratio(w, h))
        except ValueError: raise gr.Error("Resolución non válida. Usa: 1920x1080")
    
    cmd_final = [FFMPEG_PATH, "-framerate", str(input_ffmpeg_rate), "-i", str(frame_source)]
    tracks, vols = [], []
    if audio_track_1_tempfile:
        t = process_audio_track(1, audio_track_1_tempfile, fade_in_1, fade_out_1, final_video_duration, speed, preserve_pitch)
        if t: tracks.append(t); vols.append(vol_track_1)
    if audio_track_2_tempfile:
        t = process_audio_track(2, audio_track_2_tempfile, fade_in_2, fade_out_2, final_video_duration, speed, preserve_pitch)
        if t: tracks.append(t); vols.append(vol_track_2)
    if process_audio and is_conventional_video and vol_original > 0:
        speed_filter = f"atempo={speed:.4f}"
        if preserve_pitch and verify_audio_filters()[1]: speed_filter = f"rubberband=tempo={speed:.4f}"
        orig_audio = cache_path / "original_audio_processed.wav"
        try:
            subprocess.run([FFMPEG_PATH, "-i", str(input_path), "-vn", "-af", speed_filter, "-acodec", "pcm_s16le", "-y", str(orig_audio)], check=True, capture_output=True)
            tracks.append(str(orig_audio)); vols.append(vol_original)
        except subprocess.CalledProcessError: logger.warning("Non se puido procesar o audio orixinal.")
    output_dir = Path(output_path).parent
    if output_dir != Path(): output_dir.mkdir(parents=True, exist_ok=True)
    if not tracks: cmd_final.append("-an")
    else:
        for t in tracks: cmd_final.extend(["-i", t])
        mix_inputs = "".join(f"[{i+1}:a]" for i in range(len(tracks)))
        mix_filter = f"{mix_inputs}amix=inputs={len(tracks)}:duration=first[aout]" if len(tracks) > 1 else "[a1]volume=1.0[aout]"
        vol_filters = ";".join(f"[{i+1}:a]volume={vol:.3f}[a{i+1}]" for i, vol in enumerate(vols))
        cmd_final.extend(["-filter_complex", f"{vol_filters};{mix_filter}", "-map", "0:v", "-map", "[aout]"])
    if vf_filters: cmd_final.extend(["-vf", ",".join(vf_filters)])
    encoder = ["-c:v", "h264_nvenc", "-preset", "p5", "-qp", str(int(quality_crf))] if use_gpu and detect_gpu() else ["-c:v", "libx264", "-preset", "fast", "-crf", str(int(quality_crf))]
    cmd_final.extend(encoder + ["-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-y", output_path, "-shortest"])
    logger.info(f"Executando FFmpeg: {' '.join(cmd_final)}")
    try:
        result = subprocess.run(cmd_final, check=True, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0: raise gr.Error(f"Erro de FFmpeg:\n{result.stderr.decode()}")
    except Exception as e: raise gr.Error(f"O procesamento de vídeo fallou: {e}")
    return output_path, input_info, get_video_info(output_path)

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Video Converter Pro+") as demo:
        gr.Markdown("## 🎬 Video Converter Pro+ (Optimizado)")
        with gr.Row():
            with gr.Column(scale=2):
                input_file = gr.File(label="Ficheiro de entrada", file_types=['video', 'image'], file_count="single")
                output_path = gr.Textbox(label="Ruta de saída", value="output.mp4", placeholder="Introduce o nome do ficheiro de saída...")
                with gr.Accordion("💾 Xestión de Axustes", open=True):
                    preset_dropdown = gr.Dropdown(label="Cargar axuste gardado", choices=get_preset_choices(), interactive=True)
                    with gr.Row():
                        load_btn = gr.Button("Cargar")
                        delete_btn = gr.Button("Eliminar", variant="stop")
                    with gr.Row():
                        new_preset_name = gr.Textbox(label="Nome do novo axuste", placeholder="Escribe un nome e preme Gardar...")
                        save_btn = gr.Button("Gardar Axustes Actuais")
                with gr.Accordion("⚙️ Opcións principais", open=True):
                    with gr.Row():
                        speed = gr.Slider(label="Velocidade", minimum=0.25, maximum=4.0, value=1.0, step=0.05)
                        quality = gr.Slider(label="Calidade (CRF/QP)", minimum=18, maximum=35, value=23, step=1)
                    gpu = gr.Checkbox(label="Usar GPU (NVIDIA NVENC)", value=True)
                with gr.Accordion("📐 Resolución e FPS", open=False):
                    with gr.Row():
                        scale_enabled = gr.Checkbox(label="Activar escalado", value=False)
                        scale_factor = gr.Slider(label="Factor de escalado (%)", minimum=25, maximum=400, value=100, step=5)
                    fixed_res_enabled = gr.Checkbox(label="Recortar a resolución fixa", value=False)
                    fixed_res_value = gr.Textbox(label="Resolución (ex., 1920x1080)", placeholder="anchuraXaltura")
                    with gr.Row():
                        interpolate = gr.Checkbox(label="Interpolar (mellor cámara lenta)", value=False)
                        custom_fps = gr.Checkbox(label="Usar FPS personalizado", value=False)
                    fps_value = gr.Slider(label="FPS obxectivo", minimum=5, maximum=120, value=25.0, step=1) # FIX: Value as float
                with gr.Accordion("🎨 Corrección de cor", open=False):
                    color_preset = gr.Dropdown(label="Predefinición de cor", choices=list(COLOR_PRESETS.keys()), value="Normal")
                    with gr.Row():
                        brightness = gr.Slider(label="Brillo", minimum=-0.3, maximum=0.3, value=0.0, step=0.01)
                        contrast = gr.Slider(label="Contraste", minimum=0.7, maximum=1.5, value=1.0, step=0.01)
                        saturation = gr.Slider(label="Saturación", minimum=0.5, maximum=2.0, value=1.0, step=0.01)
                    with gr.Row():
                        sharp = gr.Slider(label="Enfoque", minimum=0.0, maximum=10.0, value=0.0, step=0.1)
                        blur = gr.Slider(label="Desenfoque", minimum=0.0, maximum=10.0, value=0.0, step=0.1)
                        gamma = gr.Slider(label="Gamma", minimum=0.1, maximum=10.0, value=1.0, step=0.1)
                    preview_btn = gr.Button("🔍 Previsualizar cambios de cor")
                with gr.Accordion("🎵 Opcións de audio", open=False):
                    with gr.Row():
                        process_audio = gr.Checkbox(label="Procesar audio orixinal", value=True)
                        preserve_pitch = gr.Checkbox(label="Preservar ton", value=True)
                    with gr.Accordion("Pista de audio 1 (ex., música)", open=False):
                        audio_track_1 = gr.File(label="Subir pista de audio 1", file_types=['audio'])
                        with gr.Row():
                            fade_in_1 = gr.Slider(label="Entrada gradual (s)", minimum=0.0, maximum=10.0, value=1.0, step=0.1)
                            fade_out_1 = gr.Slider(label="Saída gradual (s)", minimum=0.0, maximum=10.0, value=1.0, step=0.1)
                    with gr.Accordion("Pista de audio 2 (ex., efectos de son)", open=False):
                        audio_track_2 = gr.File(label="Subir pista de audio 2", file_types=['audio'])
                        with gr.Row():
                            fade_in_2 = gr.Slider(label="Entrada gradual (s)", minimum=0.0, maximum=10.0, value=1.0, step=0.1)
                            fade_out_2 = gr.Slider(label="Saída gradual (s)", minimum=0.0, maximum=10.0, value=1.0, step=0.1)
                    with gr.Accordion("🎚️ Mesturador de audio", open=False):
                        vol_original = gr.Slider(label="Volume do audio orixinal", minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                        vol_track_1 = gr.Slider(label="Volume da pista 1", minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                        vol_track_2 = gr.Slider(label="Volume da pista 2", minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                
                process_btn = gr.Button("🚀 Procesar vídeo", variant="primary")
                cancel_btn = gr.Button("❌ Cancelar Procesamento")
                quit_btn = gr.Button("🔴 Pechar Aplicación")
            
            with gr.Column(scale=3):
                result_video = gr.Video(label="Vídeo de saída", interactive=False, height=540)
                with gr.Row():
                    input_info = gr.Markdown(label="Información do ficheiro de entrada")
                    output_info = gr.Markdown(label="Información do ficheiro de saída")
                with gr.Row(visible=True) as preview_row:
                    before_preview = gr.Image(label="Antes")
                    after_preview = gr.Image(label="Despois")

        preset_components = [
            speed, quality, gpu, scale_enabled, scale_factor, fixed_res_enabled,
            fixed_res_value, interpolate, custom_fps, fps_value, brightness,
            contrast, saturation, sharp, blur, gamma, process_audio, preserve_pitch,
            fade_in_1, fade_out_1, fade_in_2, fade_out_2, vol_original,
            vol_track_1, vol_track_2
        ]
        
        # FIX: pasamos fps_value como un input separado para que non se mesture
        all_process_inputs = [
            input_file, output_path, speed, quality, gpu, scale_enabled, scale_factor,
            custom_fps, fps_value, interpolate, fixed_res_enabled, fixed_res_value,
            process_audio, preserve_pitch, audio_track_1, fade_in_1, fade_out_1,
            audio_track_2, fade_in_2, fade_out_2, vol_original, vol_track_1,
            vol_track_2, brightness, contrast, saturation, sharp, blur, gamma
        ]
        
        preview_btn.click(fn=preview_color_correction, inputs=[input_file, brightness, contrast, saturation, sharp, blur, gamma], outputs=[before_preview, after_preview])
        color_preset.change(fn=apply_color_preset, inputs=color_preset, outputs=[brightness, contrast, saturation, sharp, blur, gamma])
        process_event = process_btn.click(fn=process_video, inputs=all_process_inputs, outputs=[result_video, input_info, output_info])
        cancel_btn.click(fn=None, inputs=None, outputs=None, cancels=[process_event])
        save_btn.click(fn=save_preset, inputs=[new_preset_name] + preset_components, outputs=preset_dropdown)
        load_btn.click(fn=load_preset, inputs=preset_dropdown, outputs=preset_components)
        delete_btn.click(fn=delete_preset, inputs=preset_dropdown, outputs=preset_dropdown)
        
        def shutdown_app():
            os._exit(0)
        quit_btn.click(fn=shutdown_app)
    
    return demo

if __name__ == "__main__":
    verify_tools()
    demo = create_interface()
    demo.launch(inbrowser=True, server_name="0.0.0.0")