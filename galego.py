# --- Video Converter Pro+ (Versión Final con Correccións Cirúrxicas) ---

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
CACHE_DIR = Path(".video_converter_cache")
TEMP_DIR = Path(tempfile.gettempdir()) / "video_converter_temp"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Color correction presets
COLOR_PRESETS = {
    "Normal": {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0},
    "Vibrant": {"brightness": 0.02, "contrast": 1.15, "saturation": 1.25},
    "Soft": {"brightness": 0.05, "contrast": 0.95, "saturation": 0.9},
    "High Contrast": {"brightness": 0.0, "contrast": 1.25, "saturation": 1.1},
    "Desaturated": {"brightness": 0.0, "contrast": 1.05, "saturation": 0.7},
    "Warm": {"brightness": 0.03, "contrast": 1.05, "saturation": 1.15},
    "Cool": {"brightness": -0.02, "contrast": 1.1, "saturation": 1.05}
}

def verify_tools():
    """Verifica que FFmpeg e FFprobe estean dispoñibles."""
    if not all([FFMPEG_PATH, FFPROBE_PATH]):
        raise gr.Error("FFmpeg e FFprobe son necesarios. Asegúrate de que estean na ruta do teu sistema (PATH).")
    CACHE_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    logger.info("Ferramentas verificadas correctamente")

def get_video_duration(file_path):
    """Obtén a duración dun ficheiro de vídeo."""
    try:
        cmd = [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        return float(result.stdout.strip())
    except: return 0.0

def get_audio_duration(file_path):
    """Obtén a duración dun ficheiro de audio."""
    try:
        cmd = [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        return float(result.stdout.strip())
    except: return 0.0

def get_video_info(file_path):
    """Obtén información detallada dun ficheiro multimedia."""
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
    """Detecta se o codificador h264_nvenc de NVIDIA está dispoñible."""
    try:
        result = subprocess.run([FFMPEG_PATH, "-encoders"], capture_output=True, text=True, check=True, encoding='utf-8')
        return "h264_nvenc" in result.stdout
    except: return False

def verify_audio_filters():
    """Verifica a dispoñibilidade dos filtros de audio atempo e rubberband."""
    try:
        result = subprocess.run([FFMPEG_PATH, "-filters"], capture_output=True, text=True, check=True, encoding='utf-8')
        return "atempo" in result.stdout, "rubberband" in result.stdout
    except: return False, False

def get_original_fps(file_path):
    """Obtén os FPS orixinais dun ficheiro multimedia."""
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
    """Escala un único fotograma."""
    input_path, output_path, scale_factor = args
    try:
        with Image.open(input_path) as img:
            new_width = int(img.width * scale_factor / 100)
            new_height = int(img.height * scale_factor / 100)
            img.resize((new_width, new_height), Image.Resampling.LANCZOS).save(output_path, optimize=True)
    except Exception as e: logger.error(f"Erro ao escalar {input_path}: {e}")

def generate_crop_filter_for_aspect_ratio(w, h):
    """Xera un filtro de recorte para manter a relación de aspecto."""
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"

def validate_color_parameters(b, c, s):
    """Valida e limita os parámetros de cor."""
    return max(-0.3, min(0.3, b)), max(0.7, min(1.5, c)), max(0.5, min(2.0, s))

def preview_color_correction(input_tempfile, b, c, s):
    """Xera unha previsualización dos cambios de cor."""
    if not input_tempfile: raise gr.Error("Por favor, sube un ficheiro primeiro.")
    input_path = input_tempfile.name
    b, c, s = validate_color_parameters(b, c, s)
    original_frame, corrected_frame = TEMP_DIR / f"original_{time.time()}.png", TEMP_DIR / f"corrected_{time.time()}.png"
    try:
        with Image.open(input_path) as img: img.seek(0); img.copy().convert("RGB").save(original_frame)
    except:
        subprocess.run([FFMPEG_PATH, "-i", input_path, "-vframes", "1", "-q:v", "2", "-y", str(original_frame)], check=True, capture_output=True)
    
    # CORRECCIÓN do erro 'NameError' para consistencia
    filters = []
    if b != 0.0:
        filters.append(f"brightness={b:.3f}")
    if c != 1.0:
        filters.append(f"contrast={c:.3f}")
    if s != 1.0:
        filters.append(f"saturation={s:.3f}")

    if not filters: shutil.copy(original_frame, corrected_frame)
    else:
        try:
            subprocess.run([FFMPEG_PATH, "-i", str(original_frame), "-vf", "eq=" + ":".join(filters), "-y", str(corrected_frame)], check=True, capture_output=True)
        except subprocess.CalledProcessError as e: raise gr.Error(f"A corrección de cor fallou: {e.stderr.decode()}")
    return str(original_frame), str(corrected_frame)

def apply_color_preset(name):
    """Aplica unha predefinición de cor."""
    p = COLOR_PRESETS.get(name, COLOR_PRESETS["Normal"])
    return p["brightness"], p["contrast"], p["saturation"]

def process_audio_track(track_id, temp_file, fade_in, fade_out, target_duration, speed, preserve_pitch):
    """Procesa unha pista de audio individual."""
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
    try: subprocess.run(cmd, check=True, capture_output=True, text=True); return str(cached_path)
    except subprocess.CalledProcessError as e: logger.error(f"Procesamento de audio {track_id} fallou: {e}"); return None

def process_video(input_tempfile, output_path, speed=1.0, quality_crf=23, use_gpu=False,
                  scale_enabled=False, scale_factor=100, use_custom_fps=False, target_fps=25.0,
                  interpolate=False, use_fixed_res=False, fixed_resolution="", process_audio=False,
                  preserve_pitch=False,
                  audio_track_1_tempfile=None, fade_in_1=1.0, fade_out_1=1.0,
                  audio_track_2_tempfile=None, fade_in_2=1.0, fade_out_2=1.0,
                  vol_original=1.0, vol_track_1=1.0, vol_track_2=1.0,
                  brightness=0.0, contrast=1.0, saturation=1.0,
                  progress=gr.Progress(track_tqdm=True)):
    """Función principal para procesar o vídeo."""
    verify_tools()
    if not input_tempfile: raise gr.Error("Por favor, sube un ficheiro.")
    input_path = Path(input_tempfile.name)
    brightness, contrast, saturation = validate_color_parameters(brightness, contrast, saturation)
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

    # --- ESTRUTURA DE COMANDO FFmpeg ROBUSTA E DEFINITIVA ---

    # 1. Comando base coa entrada de vídeo
    final_video_duration = original_duration / speed
    input_ffmpeg_rate = original_fps * speed
    cmd_final = [FFMPEG_PATH, "-framerate", str(input_ffmpeg_rate), "-i", str(frame_source)]
    
    # 2. Recoller pistas de audio procesadas
    tracks, vols = [], []
    if process_audio and is_conventional_video and vol_original > 0:
        speed_filter = f"atempo={speed:.4f}"
        if preserve_pitch and verify_audio_filters()[1]: speed_filter = f"rubberband=tempo={speed:.4f}"
        orig_audio = cache_path / "original_audio_processed.wav"
        try:
            subprocess.run([FFMPEG_PATH, "-i", str(input_path), "-vn", "-af", speed_filter, "-acodec", "pcm_s16le", "-y", str(orig_audio)], check=True, capture_output=True)
            tracks.append(str(orig_audio)); vols.append(vol_original)
        except subprocess.CalledProcessError: logger.warning("Non se puido procesar o audio orixinal.")
    
    if audio_track_1_tempfile:
        t = process_audio_track(1, audio_track_1_tempfile, fade_in_1, fade_out_1, final_video_duration, speed, preserve_pitch)
        if t: tracks.append(t); vols.append(vol_track_1)
    if audio_track_2_tempfile:
        t = process_audio_track(2, audio_track_2_tempfile, fade_in_2, fade_out_2, final_video_duration, speed, preserve_pitch)
        if t: tracks.append(t); vols.append(vol_track_2)

    # Engadir todas as pistas de audio como entradas
    for t in tracks:
        cmd_final.extend(["-i", t])

    # 3. Construír a cadea de filtros de VÍDEO para -vf
    vf_filters = []
    if speed != 1.0: vf_filters.append(f"setpts={1.0/speed:.4f}*PTS")
    if speed < 1.0 and interpolate: vf_filters.append(f"minterpolate=fps={target_fps:.2f}")
    elif use_custom_fps and abs(target_fps - (original_fps * speed)) > 0.01: vf_filters.append(f"fps={target_fps:.2f}")
    
    # --- CORRECCIÓN DO ERRO 'NameError' ---
    color_filters_list = []
    if brightness != 0.0:
        color_filters_list.append(f"brightness={brightness:.3f}")
    if contrast != 1.0:
        color_filters_list.append(f"contrast={contrast:.3f}")
    if saturation != 1.0:
        color_filters_list.append(f"saturation={saturation:.3f}")

    if color_filters_list: vf_filters.append("eq=" + ":".join(color_filters_list))
    if use_fixed_res and 'x' in fixed_resolution:
        try: w, h = map(int, fixed_resolution.lower().split('x')); vf_filters.append(generate_crop_filter_for_aspect_ratio(w, h))
        except ValueError: raise gr.Error("Resolución non válida. Usa: 1920x1080")
    
    if vf_filters:
        cmd_final.extend(["-vf", ",".join(vf_filters)])

    # 4. Construír os filtros de AUDIO e os mapeos
    if not tracks:
        cmd_final.append("-an") # Sen audio
    elif len(tracks) == 1:
        # Caso simple: 1 pista de audio. Usamos -af para o volume.
        cmd_final.extend(["-af", f"volume={vols[0]:.3f}"])
        cmd_final.extend(["-map", "0:v", "-map", "1:a"])
    else:
        # Caso complexo: >1 pista de audio. Usamos -filter_complex SÓ para o audio.
        vol_filters = ";".join(f"[{i+1}:a]volume={vols[i]:.3f}[a{i+1}]" for i in range(len(tracks)))
        mix_inputs = "".join(f"[a{i+1}]" for i in range(len(tracks)))
        filter_complex_str = f"{vol_filters};{mix_inputs}amix=inputs={len(tracks)}:duration=first[aout]"
        cmd_final.extend(["-filter_complex", filter_complex_str])
        cmd_final.extend(["-map", "0:v", "-map", "[aout]"])

    # 5. Engadir opcións de codificación e saída
    output_dir = Path(output_path).parent
    if output_dir != Path(): output_dir.mkdir(parents=True, exist_ok=True)
    encoder = ["-c:v", "h264_nvenc", "-preset", "p5", "-qp", str(int(quality_crf))] if use_gpu and detect_gpu() else ["-c:v", "libx264", "-preset", "fast", "-crf", str(int(quality_crf))]
    cmd_final.extend(encoder + ["-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest", "-y", output_path])
    
    logger.info(f"Executando FFmpeg: {' '.join(cmd_final)}")
    try:
        result = subprocess.run(cmd_final, check=True, capture_output=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        logger.error(f"O procesamento de vídeo fallou. Stderr:\n{e.stderr}")
        raise gr.Error(f"O procesamento de vídeo fallou:\n\n{e.stderr}")
    except Exception as e:
        logger.error(f"Ocorreu un erro inesperado: {e}")
        raise gr.Error(f"O procesamento de vídeo fallou por un erro inesperado: {e}")
        
    return output_path, input_info, get_video_info(output_path)

def create_interface():
    """Crea a interface de usuario con Gradio."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Video Converter Pro+") as demo:
        gr.Markdown("## 🎬 Video Converter Pro+ (Solución Definitiva)")
        with gr.Row():
            with gr.Column(scale=2):
                input_file = gr.File(label="Ficheiro de entrada", file_types=['video', 'image'], file_count="single")
                output_path = gr.Textbox(label="Ruta de saída", value="output.mp4", placeholder="Introduce o nome do ficheiro de saída...")
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
                    fps_value = gr.Slider(label="FPS obxectivo", minimum=5, maximum=120, value=25, step=1)
                with gr.Accordion("🎨 Corrección de cor", open=False):
                    color_preset = gr.Dropdown(label="Predefinición de cor", choices=list(COLOR_PRESETS.keys()), value="Normal")
                    with gr.Row():
                        brightness = gr.Slider(label="Brillo", minimum=-0.3, maximum=0.3, value=0.0, step=0.01)
                        contrast = gr.Slider(label="Contraste", minimum=0.7, maximum=1.5, value=1.0, step=0.01)
                        saturation = gr.Slider(label="Saturación", minimum=0.5, maximum=2.0, value=1.0, step=0.01)
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
                quit_btn = gr.Button("🔴 Pechar Aplicación")
            
            with gr.Column(scale=3):
                result_video = gr.Video(label="Vídeo de saída", interactive=False, height=540)
                with gr.Row():
                    input_info = gr.Markdown(label="Información do ficheiro de entrada")
                    output_info = gr.Markdown(label="Información do ficheiro de saída")
                with gr.Row(visible=True) as preview_row:
                    before_preview = gr.Image(label="Antes")
                    after_preview = gr.Image(label="Despois")
        
        # --- Event handlers ---
        preview_btn.click(fn=preview_color_correction, inputs=[input_file, brightness, contrast, saturation], outputs=[before_preview, after_preview])
        color_preset.change(fn=apply_color_preset, inputs=color_preset, outputs=[brightness, contrast, saturation])
        
        all_inputs = [
            input_file, output_path, speed, quality, gpu, scale_enabled, scale_factor, custom_fps, fps_value,
            interpolate, fixed_res_enabled, fixed_res_value, process_audio, preserve_pitch,
            audio_track_1, fade_in_1, fade_out_1, audio_track_2, fade_in_2, fade_out_2,
            vol_original, vol_track_1, vol_track_2, brightness, contrast, saturation
        ]
        process_btn.click(fn=process_video, inputs=all_inputs, outputs=[result_video, input_info, output_info])
        
        # --- COMPORTAMENTO DE PECHE RESTAURADO E CORRECTO ---
        def shutdown_app():
            """
            Pecha a aplicación forzando a saída do proceso.
            Este é o método que funcionaba correctamente no teu contorno.
            """
            logger.info("Recibida solicitude de peche forzado. Pechando a aplicación...")
            os._exit(0)

        quit_btn.click(fn=shutdown_app)
    
    return demo

if __name__ == "__main__":
    verify_tools()
    demo = create_interface()
    demo.launch(inbrowser=True, server_name="0.0.0.0")
