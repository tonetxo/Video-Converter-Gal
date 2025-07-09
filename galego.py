# --- Video Converter Pro+ (Complete Optimized Version) ---

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
    """Verify that required tools are available"""
    if not all([FFMPEG_PATH, FFPROBE_PATH]):
        raise gr.Error("FFmpeg e FFprobe son necesarios. Asegúrate de que estean na ruta do teu sistema (PATH).")
    
    # Create necessary directories
    CACHE_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    
    logger.info("Ferramentas verificadas correctamente")

def get_video_duration(file_path):
    """Get video duration using ffprobe"""
    try:
        cmd = [
            FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Non se puido obter a duración para {file_path}: {e}")
        return 0.0

def get_audio_duration(file_path):
    """Get audio duration using ffprobe"""
    try:
        cmd = [
            FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Non se puido obter a duración do audio para {file_path}: {e}")
        return 0.0

def get_video_info(file_path):
    """Get comprehensive video information"""
    if not file_path or not Path(file_path).exists():
        return "O ficheiro non existe ou a ruta non é válida."
    
    try:
        # Try with ffprobe first
        cmd = [
            FFPROBE_PATH, "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(file_path)
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        data = json.loads(result.stdout)
        
        video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
        
        if video_stream and video_stream.get('width', 0) > 0 and video_stream.get('height', 0) > 0:
            info = []
            info.append(f"**Formato:** {data['format'].get('format_long_name', 'N/A')}")
            
            # Calculate FPS
            fps_str = video_stream.get('r_frame_rate', '0/1')
            try:
                num, den = map(int, fps_str.split('/'))
                fps = f"{num / den:.2f}" if den > 0 else "0"
            except:
                fps = "N/A"
            
            info.append(f"**Resolución:** {video_stream.get('width', 'N/A')}x{video_stream.get('height', 'N/A')}")
            info.append(f"**FPS:** {fps}")
            info.append(f"**Códec de vídeo:** {video_stream.get('codec_name', 'N/A')}")
            
            # Frame count
            frames = video_stream.get('nb_frames', 'N/A')
            if frames != 'N/A':
                info.append(f"**Número de fotogramas:** {frames}")
            
            # Audio info
            audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
            if audio_stream:
                info.append(f"**Códec de audio:** {audio_stream.get('codec_name', 'N/A')}")
                sample_rate = audio_stream.get('sample_rate', 'N/A')
                if sample_rate != 'N/A':
                    info.append(f"**Frecuencia de mostraxe:** {sample_rate} Hz")
            else:
                info.append("**Audio:** Sen pista de audio")
            
            # Duration
            duration = float(data['format'].get('duration', 0))
            info.append(f"**Duración:** {duration:.2f} segundos")
            
            # Bitrate
            bitrate = data['format'].get('bit_rate', 'N/A')
            if bitrate != 'N/A':
                bitrate_mbps = int(bitrate) / 1_000_000
                info.append(f"**Taxa de bits:** {bitrate_mbps:.2f} Mbps")
            
            return "\n".join(info)
            
    except Exception as e:
        logger.warning(f"ffprobe fallou para {file_path}: {e}")
    
    # Fallback to PIL for image files
    try:
        with Image.open(file_path) as img:
            info = []
            w, h = img.size
            n_frames = getattr(img, 'n_frames', 1)
            duration_ms_frame = img.info.get('duration', 40)
            total_duration_s = (n_frames * duration_ms_frame) / 1000.0 if n_frames > 1 else 0
            fps = n_frames / total_duration_s if total_duration_s > 0 else 0
            
            info.append(f"**Formato:** Imaxe animada ({img.format})")
            info.append(f"**Resolución:** {w}x{h}")
            info.append(f"**FPS:** {fps:.2f}")
            info.append(f"**Códec de vídeo:** {img.format.lower()}")
            info.append(f"**Número de fotogramas:** {n_frames}")
            info.append("**Audio:** Non aplicable")
            info.append(f"**Duración:** {total_duration_s:.2f} segundos")
            
            return "\n".join(info)
            
    except Exception as e:
        logger.error(f"Non se puido ler o ficheiro {file_path}: {e}")
        return f"Non se puido ler o ficheiro con ffprobe ou PIL.\nErro: {e}"

def detect_gpu():
    """Detect if NVIDIA GPU acceleration is available"""
    try:
        result = subprocess.run([FFMPEG_PATH, "-encoders"], capture_output=True, text=True, check=True, encoding='utf-8')
        has_nvenc = "h264_nvenc" in result.stdout
        logger.info(f"Aceleración GPU dispoñible: {has_nvenc}")
        return has_nvenc
    except Exception as e:
        logger.warning(f"Non se puido detectar a GPU: {e}")
        return False

def verify_audio_filters():
    """Verify available audio filters"""
    try:
        result = subprocess.run([FFMPEG_PATH, "-filters"], capture_output=True, text=True, check=True, encoding='utf-8')
        has_atempo = "atempo" in result.stdout
        has_rubberband = "rubberband" in result.stdout
        logger.info(f"Filtros de audio - atempo: {has_atempo}, rubberband: {has_rubberband}")
        return has_atempo, has_rubberband
    except Exception as e:
        logger.warning(f"Non se puideron verificar os filtros de audio: {e}")
        return False, False

def get_original_fps(file_path):
    """Get original FPS from video file"""
    try:
        cmd_fps = [
            FFPROBE_PATH, "-v", "error", "-select_streams", "v:0",
            "-of", "default=noprint_wrappers=1:nokey=1",
            "-show_entries", "stream=r_frame_rate", str(file_path)
        ]
        fps_str = subprocess.run(cmd_fps, check=True, capture_output=True, text=True, encoding='utf-8').stdout.strip()
        num, den = map(int, fps_str.split('/'))
        return num / den if den > 0 else 25.0
    except Exception:
        # Fallback for images
        try:
            with Image.open(file_path) as img:
                n_frames = getattr(img, 'n_frames', 1)
                duration_info = img.info.get('duration', 40)
                if n_frames == 0 or duration_info <= 0:
                    return 25.0
                return n_frames / ((n_frames * duration_info) / 1000.0)
        except Exception:
            return 25.0

def scale_frame(args):
    """Scale a single frame"""
    input_path, output_path, scale_factor = args
    try:
        with Image.open(input_path) as img:
            original_width, original_height = img.size
            new_width = int(original_width * scale_factor / 100)
            new_height = int(original_height * scale_factor / 100)
            
            # Use high-quality resampling
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(output_path, optimize=True)
        return True
    except Exception as e:
        logger.error(f"Erro ao escalar {input_path}: {e}")
        return False

def generate_crop_filter_for_aspect_ratio(target_w, target_h):
    """Generate FFmpeg filter for cropping to specific aspect ratio"""
    return f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"

def validate_color_parameters(brightness, contrast, saturation):
    """Validate and clamp color correction parameters"""
    # Clamp brightness to reasonable range
    brightness = max(-0.3, min(0.3, brightness))
    
    # Clamp contrast to reasonable range
    contrast = max(0.7, min(1.5, contrast))
    
    # Clamp saturation to reasonable range
    saturation = max(0.5, min(2.0, saturation))
    
    return brightness, contrast, saturation

def preview_color_correction(input_tempfile, brightness, contrast, saturation):
    """Preview color correction on a single frame"""
    if not input_tempfile:
        raise gr.Error("Por favor, sube un ficheiro primeiro para previsualizar.")
    
    input_path = input_tempfile.name
    
    # Validate parameters
    brightness, contrast, saturation = validate_color_parameters(brightness, contrast, saturation)
    
    # Create temporary files
    original_frame = TEMP_DIR / f"original_{int(time.time())}.png"
    corrected_frame = TEMP_DIR / f"corrected_{int(time.time())}.png"
    
    try:
        # Extract frame
        if str(input_path).lower().endswith(('.webp', '.gif', '.png', '.jpg', '.jpeg')):
            with Image.open(input_path) as img:
                img.seek(0)
                img.copy().convert("RGB").save(original_frame)
        else:
            cmd_extract = [
                FFMPEG_PATH, "-i", input_path, "-vframes", "1", "-q:v", "2", 
                "-y", str(original_frame)
            ]
            subprocess.run(cmd_extract, check=True, capture_output=True)
    except Exception as e:
        raise gr.Error(f"Non se puido extraer o fotograma do ficheiro: {e}")
    
    # Apply color correction
    color_filters = []
    if brightness != 0.0:
        color_filters.append(f"brightness={brightness:.3f}")
    if contrast != 1.0:
        color_filters.append(f"contrast={contrast:.3f}")
    if saturation != 1.0:
        color_filters.append(f"saturation={saturation:.3f}")
    
    if not color_filters:
        shutil.copy(original_frame, corrected_frame)
    else:
        eq_filter = "eq=" + ":".join(color_filters)
        cmd_color = [
            FFMPEG_PATH, "-i", str(original_frame), "-vf", eq_filter, 
            "-y", str(corrected_frame)
        ]
        try:
            subprocess.run(cmd_color, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise gr.Error(f"A corrección de cor fallou: {e}")
    
    return str(original_frame), str(corrected_frame)

def apply_color_preset(preset_name):
    """Apply a color correction preset"""
    if preset_name in COLOR_PRESETS:
        preset = COLOR_PRESETS[preset_name]
        return preset["brightness"], preset["contrast"], preset["saturation"]
    return 0.0, 1.0, 1.0

def process_audio_track(track_id, temp_file, fade_in, fade_out, target_duration, speed_factor, preserve_pitch):
    """Process an audio track with speed adjustment"""
    if not temp_file:
        return None
    
    track_path = temp_file.name
    track_mtime = Path(track_path).stat().st_mtime
    
    # Create unique cache key based on parameters
    cache_key = hashlib.sha256(
        f"{track_path}-{track_mtime}-{fade_in}-{fade_out}-{target_duration}-{speed_factor}-{preserve_pitch}".encode()
    ).hexdigest()
    
    cached_audio_path = CACHE_DIR / f"audio_{track_id}_{cache_key}.aac"
    
    if cached_audio_path.exists():
        logger.info(f"Usando audio en caché para a pista {track_id}")
        return str(cached_audio_path)
    
    logger.info(f"Procesando e gardando en caché o audio para a pista {track_id}")
    
    # Calculate audio speed adjustment
    original_duration = get_audio_duration(track_path)
    if original_duration > 0:
        required_speed = original_duration / target_duration
    else:
        required_speed = 1.0
    
    # Build filters
    filters = []
    
    # Speed adjustment
    if abs(required_speed - 1.0) > 0.01:
        if preserve_pitch:
            has_atempo, has_rubberband = verify_audio_filters()
            if has_rubberband:
                filters.append(f"rubberband=tempo={required_speed:.4f}")
            elif has_atempo:
                # Handle atempo limitations (0.5-2.0 range)
                current_speed = required_speed
                while current_speed > 2.0:
                    filters.append("atempo=2.0")
                    current_speed /= 2.0
                while current_speed < 0.5:
                    filters.append("atempo=0.5")
                    current_speed /= 0.5
                if abs(current_speed - 1.0) > 0.01:
                    filters.append(f"atempo={current_speed:.4f}")
            else:
                filters.append(f"atempo={required_speed:.4f}")
        else:
            filters.append(f"atempo={required_speed:.4f}")
    
    # Fade effects
    fade_out_start = max(0, target_duration - fade_out)
    filters.append(f"afade=t=in:st=0:d={fade_in:.2f}")
    filters.append(f"afade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f}")
    
    # Combine filters
    filter_str = ",".join(filters) if filters else None
    
    # Build command
    cmd = [FFMPEG_PATH, "-i", track_path]
    
    if filter_str:
        cmd.extend(["-af", filter_str])
    
    cmd.extend([
        "-t", str(target_duration), 
        "-c:a", "aac", 
        "-b:a", "192k",
        "-y", str(cached_audio_path)
    ])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return str(cached_audio_path)
    except subprocess.CalledProcessError as e:
        logger.error(f"O procesamento de audio fallou para a pista {track_id}: {e}")
        return None

def process_video(input_tempfile, output_path, speed=1.0, quality_crf=23, use_gpu=False,
                  scale_enabled=False, scale_factor=100, use_custom_fps=False, target_fps=25.0,
                  interpolate=False, use_fixed_res=False, fixed_resolution="", process_audio=False,
                  preserve_pitch=False,
                  audio_track_1_tempfile=None, fade_in_1=1.0, fade_out_1=1.0,
                  audio_track_2_tempfile=None, fade_in_2=1.0, fade_out_2=1.0,
                  vol_original=1.0, vol_track_1=1.0, vol_track_2=1.0,
                  brightness=0.0, contrast=1.0, saturation=1.0,
                  progress=gr.Progress(track_tqdm=True)):
    """Main video processing function"""
    
    verify_tools()
    
    if not input_tempfile:
        raise gr.Error("Por favor, sube un ficheiro de vídeo ou imaxe.")
    
    input_path = Path(input_tempfile.name)
    
    # Validate color parameters
    brightness, contrast, saturation = validate_color_parameters(brightness, contrast, saturation)
    
    # Get input info
    input_info = get_video_info(input_path)
    is_conventional_video = input_path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    original_fps = get_original_fps(input_path)
    original_duration = get_video_duration(input_path)
    
    if original_duration == 0:
        try:
            with Image.open(input_path) as img:
                n_frames = getattr(img, 'n_frames', 1)
                duration_ms = img.info.get('duration', 40)
                original_duration = (n_frames * duration_ms) / 1000.0
                if original_duration == 0:
                    original_duration = 5
        except:
            original_duration = 5
    
    logger.info(f"Procesando: {input_path} (Duración: {original_duration:.2f}s, FPS: {original_fps:.2f})")
    
    # Create cache key
    mtime = input_path.stat().st_mtime
    cache_key = hashlib.sha256(
        f"{input_path}-{mtime}-{scale_enabled}-{scale_factor}".encode()
    ).hexdigest()
    cache_path = CACHE_DIR / cache_key
    frame_source = cache_path / "frame-%05d.png"
    
    # Extract and cache frames
    if cache_path.exists():
        logger.info(f"Usando fotogramas en caché: {cache_path}")
    else:
        logger.info(f"Creando nova caché de fotogramas: {cache_path}")
        cache_path.mkdir(parents=True)
        
        original_frames_dir = cache_path / "original_frames"
        original_frames_dir.mkdir()
        original_frame_pattern = original_frames_dir / "frame-%05d.png"
        
        # Extract frames
        if is_conventional_video:
            cmd_extract = [
                FFMPEG_PATH, "-i", str(input_path), "-q:v", "2", 
                str(original_frame_pattern)
            ]
            subprocess.run(cmd_extract, check=True, capture_output=True)
        else:
            # Handle images/animated images
            with Image.open(input_path) as img:
                n_frames = getattr(img, 'n_frames', 1)
                for i in progress.tqdm(range(n_frames), desc="Extraendo fotogramas"):
                    img.seek(i)
                    frame_path = original_frames_dir / f"frame-{i:05d}.png"
                    img.copy().convert("RGB").save(frame_path)
        
        # Scale frames if needed
        if scale_enabled and scale_factor != 100:
            frame_files = sorted(original_frames_dir.glob("frame-*.png"))
            args_list = [
                (str(f), str(cache_path / f.name), scale_factor)
                for f in frame_files
            ]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                results = list(progress.tqdm(
                    executor.map(scale_frame, args_list),
                    total=len(args_list),
                    desc="Escalando fotogramas"
                ))
            
            shutil.rmtree(original_frames_dir)
        else:
            # Move frames to cache directory
            for frame_file in original_frames_dir.glob("frame-*.png"):
                shutil.move(str(frame_file), cache_path / frame_file.name)
            original_frames_dir.rmdir()
    
    # Build video filters
    vf_filters = []
    input_ffmpeg_rate = original_fps
    final_video_duration = original_duration / speed
    
    # Handle speed changes
    if speed < 1.0:
        if interpolate:
            vf_filters.append(f"setpts={1.0/speed:.4f}*PTS")
            vf_filters.append(f"minterpolate=fps={target_fps:.2f}")
        else:
            input_ffmpeg_rate = original_fps * speed
            if use_custom_fps and abs(target_fps - input_ffmpeg_rate) > 0.01:
                vf_filters.append(f"fps={target_fps:.2f}")
    elif speed > 1.0:
        vf_filters.append(f"setpts={1.0/speed:.4f}*PTS")
        if use_custom_fps:
            vf_filters.append(f"fps={target_fps:.2f}")
    elif use_custom_fps and abs(target_fps - original_fps) > 0.01:
        vf_filters.append(f"fps={target_fps:.2f}")
    
    # Color correction filters
    color_filters = []
    if brightness != 0.0:
        color_filters.append(f"brightness={brightness:.3f}")
    if contrast != 1.0:
        color_filters.append(f"contrast={contrast:.3f}")
    if saturation != 1.0:
        color_filters.append(f"saturation={saturation:.3f}")
    
    if color_filters:
        vf_filters.append("eq=" + ":".join(color_filters))
    
    # Resolution filter
    if use_fixed_res and 'x' in fixed_resolution:
        try:
            w, h = map(int, fixed_resolution.lower().split('x'))
            vf_filters.append(generate_crop_filter_for_aspect_ratio(w, h))
        except ValueError:
            raise gr.Error("Formato de resolución non válido. Usa o formato: 1920x1080")
    
    # Build final command
    cmd_final = [
        FFMPEG_PATH, "-framerate", str(input_ffmpeg_rate), "-i", str(frame_source)
    ]
    
    # Process audio tracks (all tracks will be adjusted to the new duration)
    tracks_to_mix = []
    volumes = []
    
    # Process track 1
    if audio_track_1_tempfile:
        track_1_processed = process_audio_track(
            1, audio_track_1_tempfile, fade_in_1, fade_out_1, 
            final_video_duration, speed, preserve_pitch
        )
        if track_1_processed:
            tracks_to_mix.append(track_1_processed)
            volumes.append(vol_track_1)
    
    # Process track 2
    if audio_track_2_tempfile:
        track_2_processed = process_audio_track(
            2, audio_track_2_tempfile, fade_in_2, fade_out_2, 
            final_video_duration, speed, preserve_pitch
        )
        if track_2_processed:
            tracks_to_mix.append(track_2_processed)
            volumes.append(vol_track_2)
    
    # Process original audio
    if process_audio and is_conventional_video and vol_original > 0:
        audio_speed_filter = f"atempo={speed:.4f}"
        if preserve_pitch:
            has_atempo, has_rubberband = verify_audio_filters()
            if has_rubberband:
                audio_speed_filter = f"rubberband=tempo={speed:.4f}"
        
        original_audio_processed = cache_path / "original_audio_processed.wav"
        cmd_audio = [
            FFMPEG_PATH, "-i", str(input_path), "-vn", "-af", audio_speed_filter,
            "-acodec", "pcm_s16le", "-y", str(original_audio_processed)
        ]
        
        try:
            subprocess.run(cmd_audio, check=True, capture_output=True)
            tracks_to_mix.append(str(original_audio_processed))
            volumes.append(vol_original)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Non se puido procesar o audio orixinal: {e}")
    
    # Create output directory
    output_dir = Path(output_path).parent
    if output_dir != Path():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle audio mixing
    if not tracks_to_mix:
        cmd_final.append("-an")
    else:
        # Add audio inputs
        for track in tracks_to_mix:
            cmd_final.extend(["-i", track])
        
        # Build filter complex for audio mixing
        filter_complex_parts = []
        filter_complex_outputs = ""
        
        for i, vol in enumerate(volumes, 1):
            filter_complex_parts.append(f"[{i}:a]volume={vol:.3f}[a{i}]")
            filter_complex_outputs += f"[a{i}]"
        
        if len(tracks_to_mix) > 1:
            filter_complex_parts.append(
                f"{filter_complex_outputs}amix=inputs={len(tracks_to_mix)}:duration=first[aout]"
            )
        else:
            # Single track - just rename the output
            filter_complex_parts[-1] = filter_complex_parts[-1].replace("[a1]", "[aout]")
        
        cmd_final.extend([
            "-filter_complex", ";".join(filter_complex_parts),
            "-map", "0:v", "-map", "[aout]"
        ])
    
    # Add video filters
    if vf_filters:
        cmd_final.extend(["-vf", ",".join(vf_filters)])
    
    # Encoder settings
    if use_gpu and detect_gpu():
        cmd_final.extend(["-c:v", "h264_nvenc", "-preset", "fast", "-qp", str(int(quality_crf))])
    else:
        cmd_final.extend(["-c:v", "libx264", "-preset", "fast", "-crf", str(int(quality_crf))])
    
    # Audio and output settings
    cmd_final.extend([
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-y", output_path, "-shortest"
    ])
    
    # Execute final command
    logger.info(f"Executando FFmpeg: {' '.join(cmd_final)}")
    
    try:
        result = subprocess.run(cmd_final, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            raise gr.Error(f"Erro de FFmpeg:\n{result.stderr}")
    except Exception as e:
        raise gr.Error(f"O procesamento de vídeo fallou: {e}")
    
    # Get output info
    output_info = get_video_info(output_path)
    
    logger.info(f"Procesamento de vídeo completado: {output_path}")
    return output_path, input_info, output_info

# --- Gradio Interface ---
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Video Converter Pro+") as demo:
        gr.Markdown("## 🎬 Video Converter Pro+ (Optimizado)")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                input_file = gr.File(
                    label="Ficheiro de entrada", 
                    file_types=['video', 'image'],
                    file_count="single"
                )
                output_path = gr.Textbox(
                    label="Ruta de saída", 
                    value="output.mp4",
                    placeholder="Introduce o nome do ficheiro de saída..."
                )
                
                # Main options
                with gr.Accordion("⚙️ Opcións principais", open=True):
                    with gr.Row():
                        speed = gr.Slider(
                            label="Velocidade", 
                            minimum=0.25, maximum=4.0, 
                            value=1.0, step=0.05
                        )
                        quality = gr.Slider(
                            label="Calidade (CRF/QP)", 
                            minimum=18, maximum=35, 
                            value=23, step=1
                        )
                    gpu = gr.Checkbox(label="Usar GPU (NVIDIA NVENC)", value=True)
                
                # Resolution and FPS options
                with gr.Accordion("📐 Resolución e FPS", open=False):
                    with gr.Row():
                        scale_enabled = gr.Checkbox(label="Activar escalado", value=False)
                        scale_factor = gr.Slider(
                            label="Factor de escalado (%)", 
                            minimum=25, maximum=400, 
                            value=100, step=5
                        )
                    
                    fixed_res_enabled = gr.Checkbox(label="Recortar a resolución fixa", value=False)
                    fixed_res_value = gr.Textbox(
                        label="Resolución (ex., 1920x1080)", 
                        placeholder="anchuraXaltura"
                    )
                    
                    with gr.Row():
                        interpolate = gr.Checkbox(label="Interpolar (mellor cámara lenta)", value=False)
                        custom_fps = gr.Checkbox(label="Usar FPS personalizado", value=False)
                    
                    fps_value = gr.Slider(
                        label="FPS obxectivo", 
                        minimum=5, maximum=120, 
                        value=25, step=1
                    )
                
                # Color correction
                with gr.Accordion("🎨 Corrección de cor", open=False):
                    color_preset = gr.Dropdown(
                        label="Predefinición de cor",
                        choices=list(COLOR_PRESETS.keys()),
                        value="Normal"
                    )
                    
                    with gr.Row():
                        brightness = gr.Slider(
                            label="Brillo", 
                            minimum=-0.3, maximum=0.3, 
                            value=0.0, step=0.01
                        )
                        contrast = gr.Slider(
                            label="Contraste", 
                            minimum=0.7, maximum=1.5, 
                            value=1.0, step=0.01
                        )
                        saturation = gr.Slider(
                            label="Saturación", 
                            minimum=0.5, maximum=2.0, 
                            value=1.0, step=0.01
                        )
                    
                    preview_btn = gr.Button("🔍 Previsualizar cambios de cor")
                
                # Audio options
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
            
            with gr.Column(scale=3):
                # Output section
                result_video = gr.Video(label="Vídeo de saída", interactive=False, height=540)
                with gr.Row():
                    input_info = gr.Markdown(label="Información do ficheiro de entrada")
                    output_info = gr.Markdown(label="Información do ficheiro de saída")
                with gr.Row(visible=True) as preview_row:
                    before_preview = gr.Image(label="Antes")
                    after_preview = gr.Image(label="Despois")
        
        # Event handlers
        preview_btn.click(
            fn=preview_color_correction,
            inputs=[input_file, brightness, contrast, saturation],
            outputs=[before_preview, after_preview]
        )
        
        # Connect color preset dropdown
        color_preset.change(
            fn=apply_color_preset,
            inputs=color_preset,
            outputs=[brightness, contrast, saturation]
        )
        
        process_btn.click(
            fn=process_video,
            inputs=[
                input_file, output_path, speed, quality, gpu, 
                scale_enabled, scale_factor, custom_fps, fps_value,
                interpolate, fixed_res_enabled, fixed_res_value, 
                process_audio, preserve_pitch,
                audio_track_1, fade_in_1, fade_out_1,
                audio_track_2, fade_in_2, fade_out_2,
                vol_original, vol_track_1, vol_track_2,
                brightness, contrast, saturation
            ],
            outputs=[result_video, input_info, output_info]
        )
    
    return demo

if __name__ == "__main__":
    verify_tools()
    demo = create_interface()
    demo.launch(debug=True)
