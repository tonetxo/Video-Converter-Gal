import gradio as gr
import subprocess
import os
import shutil
import time
import hashlib
import concurrent.futures
import logging
import tempfile
from pathlib import Path
from PIL import Image

from .config import FFMPEG_PATH, CACHE_DIR, TEMP_DIR, MAX_THREADS, translations, COLOR_PRESETS
from .utils import (
    get_video_duration, get_audio_duration, get_original_fps,
    get_video_info, detect_gpu, verify_audio_filters, get_frame_dimensions
)

# --- Global In-Memory Cache for final results ---
FINAL_VIDEO_CACHE = {}

logger = logging.getLogger(__name__)

def scale_frame(args):
    input_path, output_path, scale_factor = args
    try:
        with Image.open(input_path) as img:
            new_width = int(img.width * scale_factor / 100)
            new_height = int(img.height * scale_factor / 100)
            img.resize((new_width, new_height), Image.Resampling.LANCZOS).save(output_path, optimize=True)
    except Exception as e: logger.error(f"Error scaling {input_path}: {e}")

def generate_crop_filter_for_aspect_ratio(w, h):
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"

def validate_color_parameters(b, c, s, sharp, blur, gamma):
    return (max(-0.3, min(0.3, b)), max(0.7, min(1.5, c)), max(0.5, min(2.0, s)), max(-2.0, min(5.0, sharp)), max(0.0, min(10.0, blur)), max(0.1, min(10.0, gamma)))

def preview_color_correction_by_frame(input_tempfile, frame_number, b, c, s, sharp, blur, gamma, language="galego"):
    if not input_tempfile:
        return None, None
    
    input_path = Path(input_tempfile)
    if not input_path.exists():
        return None, None

    frame_number = int(frame_number)
    
    # Define paths for the temporary frame images
    temp_dir = TEMP_DIR / f"frames_preview_{input_path.stem}_{int(time.time())}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_frame_path = temp_dir / f"original_frame_{frame_number}.png"
    corrected_frame_path = temp_dir / f"corrected_frame_{frame_number}.png"

    # --- Extract the specific frame ---
    try:
        time.sleep(0.2)
        if not input_path.exists() or input_path.stat().st_size == 0:
            raise FileNotFoundError(f"Preview file is not ready or is empty: {input_path}")

        # The comma in the select filter needs to be escaped for ffmpeg
        cmd_extract = [
            FFMPEG_PATH, "-i", str(input_path),
            "-vf", f"select=eq(n\,{frame_number})",
            "-vframes", "1", str(original_frame_path), "-y"
        ]
        subprocess.run(cmd_extract, check=True, capture_output=True, text=True, encoding='utf-8')
    except Exception as e:
        try:
            with Image.open(input_path) as img:
                img.seek(frame_number)
                img.copy().convert("RGB").save(original_frame_path)
        except Exception as e2:
            logger.error(f"Error extracting frame {frame_number} from {input_path} with both ffmpeg and Pillow: {e} / {e2}")
            return None, None

    # --- Apply color correction to the extracted frame ---
    b, c, s, sharp, blur, gamma = validate_color_parameters(b, c, s, sharp, blur, gamma)
    
    filter_chain, eq_params = [], []
    if b != 0.0: eq_params.append(f"brightness={b:.3f}")
    if c != 1.0: eq_params.append(f"contrast={c:.3f}")
    if s != 1.0: eq_params.append(f"saturation={s:.3f}")
    if gamma != 1.0: eq_params.append(f"gamma={gamma:.3f}")
    if eq_params: filter_chain.append("eq=" + ":".join(eq_params))
    if blur > 0.0: filter_chain.append(f"boxblur=luma_radius={blur:.3f}:luma_power=1")
    if sharp != 0.0: filter_chain.append(f"unsharp=lx=5:ly=5:la={sharp:.3f}")

    if not filter_chain:
        shutil.copy(original_frame_path, corrected_frame_path)
    else:
        try:
            cmd_filter = [
                FFMPEG_PATH, "-i", str(original_frame_path),
                "-vf", ",".join(filter_chain),
                "-y", str(corrected_frame_path)
            ]
            subprocess.run(cmd_filter, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error applying color correction: {e.stderr.decode()}")
            return str(original_frame_path), None

    return str(original_frame_path), str(corrected_frame_path)

def apply_color_preset(name, language="galego"):
    p = COLOR_PRESETS.get(name, COLOR_PRESETS["Normal"])
    return p["brightness"], p["contrast"], p["saturation"], p["sharp"], p["blur"], p["gamma"]


def process_audio_track(track_id, temp_file, fade_in, fade_out, target_duration, speed, preserve_pitch, sync_to_video=False, trim_to_video=False, video_original_duration=None):
    if not temp_file: return None
    track_path, mtime = temp_file.name, Path(temp_file.name).stat().st_mtime
    if not Path(track_path).exists():
        logger.error(f"Audio track {track_path} does not exist.")
        return None
    key = hashlib.sha256(f"{track_path}-{mtime}-{fade_in}-{fade_out}-{target_duration}-{speed}-{preserve_pitch}-{sync_to_video}-{trim_to_video}-{video_original_duration}".encode()).hexdigest()
    cached_path = CACHE_DIR / f"audio_{track_id}_{key}.aac"
    if cached_path.exists(): 
        logger.info(f"Using cached audio track: {cached_path}")
        return str(cached_path)
    orig_dur = get_audio_duration(track_path)
    if orig_dur == 0:
        logger.error(f"Could not determine duration of audio track {track_path}.")
        return None
    filters = []
    
    effective_duration = video_original_duration if trim_to_video and video_original_duration is not None else (orig_dur if not sync_to_video else target_duration)
    
    if sync_to_video and abs(speed - 1.0) > 0.01:
        req_speed = orig_dur / target_duration if orig_dur > 0 else 1.0
        current_speed = req_speed
        if preserve_pitch and verify_audio_filters()[1]:
            filters.append(f"rubberband=tempo={req_speed:.4f}")
        elif preserve_pitch and verify_audio_filters()[0]:
            while current_speed > 2.0:
                filters.append("atempo=2.0")
                current_speed /= 2.0
            while current_speed < 0.5:
                filters.append("atempo=0.5")
                current_speed /= 0.5
            if abs(current_speed - 1.0) > 0.01:
                filters.append(f"atempo={current_speed:.4f}")
        else:
            filters.append(f"atempo={req_speed:.4f}")

    fade_out_start = max(0, effective_duration - fade_out)
    if fade_in > 0:
        filters.append(f"afade=t=in:st=0:d={fade_in:.2f}")
    if fade_out > 0:
        filters.append(f"afade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f}")

    cmd = [FFMPEG_PATH, "-i", track_path]
    if filters:
        cmd.extend(["-af", ",".join(filters)])
    cmd.extend(["-t", str(effective_duration), "-c:a", "aac", "-b:a", "192k", "-y", str(cached_path)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.info(f"Audio track {track_id} processed successfully: {cached_path}")
        return str(cached_path)
    except subprocess.CalledProcessError as e:
        logger.error(f"Audio track {track_id} processing failed: {e.stderr}")
        return None

def create_preview(input_tempfile):
    if not input_tempfile:
        return None

    input_path = Path(input_tempfile)
    is_conventional_video = input_path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"]

    if is_conventional_video:
        logger.info(f"Input is a standard video: {input_path}. No conversion needed for preview.")
        return str(input_path), get_video_info(input_path)

    logger.info(f"Input is an image: {input_path}. Creating video preview.")
    try:
        preview_temp_dir = TEMP_DIR / f"preview_{input_path.stem}_{int(time.time())}"
        preview_temp_dir.mkdir(parents=True, exist_ok=True)
        frame_pattern = preview_temp_dir / "frame-%05d.png"
        output_preview_mp4 = preview_temp_dir / "preview.mp4"

        with Image.open(input_path) as img:
            n_frames = getattr(img, 'n_frames', 1)
            logger.info(f"Extracting {n_frames} frames from {input_path.name}")
            for i in range(n_frames):
                img.seek(i)
                img.copy().convert("RGB").save(preview_temp_dir / f"frame-{i:05d}.png")

        original_fps = get_original_fps(input_path)
        
        # Add scale filter to ensure dimensions are even for libx264
        cmd = [
            FFMPEG_PATH,
            "-framerate", str(original_fps),
            "-i", str(frame_pattern),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y", str(output_preview_mp4)
        ]
        
        logger.info(f"Assembling preview video: {' '.join(map(str, cmd))}")
        try:
            subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True, encoding='utf-8')
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg preview assembly failed. Return code: {e.returncode}")
            logger.error(f"FFmpeg stdout: {e.stdout}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise e
        
        logger.info(f"Preview created successfully: {output_preview_mp4}")
        return str(output_preview_mp4), get_video_info(output_preview_mp4)

    except Exception as e:
        logger.error(f"Failed to create preview for {input_path}: {e}", exc_info=True)
        return None, None # Return a tuple on failure to avoid unpack error


def process_video(input_tempfile, output_path, original_video_info_str, speed=1.0, quality_crf=23, use_gpu=False,
                  scale_enabled=False, scale_factor=100, use_custom_fps=False, fps_value=25.0,
                  interpolate=False, use_fixed_res=False, fixed_resolution="", res_mode="Fit",
                  rotation="0", process_audio=False, preserve_pitch=False,
                  audio_track_1_tempfile=None, fade_in_1=1.0, fade_out_1=1.0,
                  audio_track_2_tempfile=None, fade_in_2=1.0, fade_out_2=1.0,
                  vol_original=1.0, vol_track_1=1.0, vol_track_2=1.0,
                  brightness=0.0, contrast=1.0, saturation=1.0,
                  sharp=0.0, blur=0.0, gamma=1.0, trim_to_video=False, use_shortest=False,
                  progress=gr.Progress(track_tqdm=True), language="galego"):
    if not input_tempfile: raise gr.Error(translations[language]["error_no_input_file"])
    
    input_path = Path(input_tempfile)
    if not input_path.exists(): raise gr.Error(f"{translations[language]['error_input_file_not_found']} {input_path}")

    # --- In-Memory Cache Check ---
    mtime = input_path.stat().st_mtime
    params_string = (f"{input_path}-{mtime}-{speed}-{quality_crf}-{use_gpu}-{scale_enabled}-{scale_factor}-"
                     f"{use_custom_fps}-{fps_value}-{interpolate}-{use_fixed_res}-{fixed_resolution}-{res_mode}-"
                     f"{rotation}-{process_audio}-{preserve_pitch}-{vol_original}-{vol_track_1}-{vol_track_2}-"
                     f"{brightness}-{contrast}-{saturation}-{sharp}-{blur}-{gamma}-{trim_to_video}-{use_shortest}-"
                     f"{getattr(audio_track_1_tempfile, 'name', 'None')}-{fade_in_1}-{fade_out_1}-"
                     f"{getattr(audio_track_2_tempfile, 'name', 'None')}-{fade_in_2}-{fade_out_2}")
    
    cache_key = hashlib.sha256(params_string.encode()).hexdigest()
    
    if cache_key in FINAL_VIDEO_CACHE:
        cached_result = FINAL_VIDEO_CACHE[cache_key]
        if Path(cached_result["path"]).exists():
            logger.info(f"Returning cached result for key {cache_key}")
            gr.Info(translations[language]["cache_hit_message"])
            return cached_result["path"], cached_result["original_info"], cached_result["output_info"]
        else:
            logger.warning(f"Cached file for key {cache_key} not found. Reprocessing.")
            del FINAL_VIDEO_CACHE[cache_key]

    brightness, contrast, saturation, sharp, blur, gamma = validate_color_parameters(brightness, contrast, saturation, sharp, blur, gamma)
    original_input_info = original_video_info_str
    is_conventional_video = input_path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    original_fps, original_duration = get_original_fps(input_path), get_video_duration(input_path)
    if original_duration == 0:
        try:
            with Image.open(input_path) as img:
                n_frames = getattr(img, 'n_frames', 1)
                duration_ms = img.info.get('duration', 40)
                original_duration = (n_frames * duration_ms) / 1000.0 if n_frames > 1 else 5.0
        except: original_duration = 5.0
    logger.info(f"Processing: {input_path} (Duration: {original_duration:.2f}s, FPS: {original_fps:.2f}, trim_to_video: {trim_to_video}, use_shortest: {use_shortest})")
    
    # --- Nivel 1: Caché de fotogramas brutos ---
    raw_frames_key = hashlib.sha256(f"{input_path}-{mtime}".encode()).hexdigest()
    raw_frames_path = CACHE_DIR / raw_frames_key
    if not raw_frames_path.exists() or not any(raw_frames_path.iterdir()):
        logger.info(f"No raw frames cache found for {input_path}. Extracting frames.")
        raw_frames_path.mkdir(parents=True, exist_ok=True)
        original_frame_pattern = raw_frames_path / "frame-%05d.png"
        
        if is_conventional_video: 
            try:
                subprocess.run([FFMPEG_PATH, "-i", str(input_path), "-q:v", "2", str(original_frame_pattern)], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                raise gr.Error(f"{translations[language]['error_extracting_frames']}: {e.stderr}")
        else:
            with Image.open(input_path) as img:
                num_frames = getattr(img, 'n_frames', 1)
                for i in progress.tqdm(range(num_frames), desc="Extracting frames"):
                    img.seek(i)
                    img.copy().convert("RGB").save(raw_frames_path / f"frame-{i:05d}.png")
    else:
        logger.info(f"Found raw frames cache at {raw_frames_path}")

    # --- Nivel 2: Determinar a orixe dos fotogramas (brutos ou escalados) ---
    frame_source_path = raw_frames_path
    if scale_enabled and scale_factor != 100:
        scaled_frames_key = f"scaled_{scale_factor}"
        scaled_frames_path = raw_frames_path / scaled_frames_key
        if not scaled_frames_path.exists() or not any(scaled_frames_path.iterdir()):
            logger.info(f"No scaled frames cache found for scale {scale_factor}%. Scaling now.")
            scaled_frames_path.mkdir(exist_ok=True)
            
            frame_files_to_scale = sorted(raw_frames_path.glob("frame-*.png"))
            args_list = [(str(f), str(scaled_frames_path / f.name), scale_factor) for f in frame_files_to_scale]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                list(progress.tqdm(executor.map(scale_frame, args_list), total=len(args_list), desc=f"Scaling to {scale_factor}%"))
        else:
            logger.info(f"Found scaled frames cache at {scaled_frames_path}")
        frame_source_path = scaled_frames_path

    frame_source = frame_source_path / "frame-%05d.png"
    
    frame_files = sorted(frame_source_path.glob("frame-*.png"))
    if not frame_files:
        raise gr.Error(f"{translations[language]['error_no_frames_in_cache']}: {frame_source_path}")
    frame_width, frame_height = get_frame_dimensions(frame_files[0])
    if frame_width == 0 or frame_height == 0:
        raise gr.Error(f"{translations[language]['error_cannot_determine_frame_dimensions']}: {frame_files[0]}")
    
    vf_filters, final_video_duration = [], original_duration / speed
    
    if speed != 1.0:
        input_ffmpeg_rate = original_fps * speed
    else:
        input_ffmpeg_rate = original_fps
    
    if speed < 1.0 and interpolate:
        vf_filters.append(f"minterpolate=fps={float(fps_value):.2f}")
    elif use_custom_fps and abs(float(fps_value) - (input_ffmpeg_rate if speed != 1.0 else original_fps)) > 0.01:
        vf_filters.append(f"fps={float(fps_value):.2f}")

    rotation_value = {"0": 0, "90": 1, "180": 2, "270": 3}.get(rotation, 0)
    if rotation_value != 0:
        vf_filters.append(f"transpose={rotation_value}")

    eq_params = []
    if brightness != 0.0: eq_params.append(f"brightness={brightness:.3f}")
    if contrast != 1.0: eq_params.append(f"contrast={contrast:.3f}")
    if saturation != 1.0: eq_params.append(f"saturation={saturation:.3f}")
    if gamma != 1.0: eq_params.append(f"gamma={gamma:.3f}")
    if eq_params: vf_filters.append("eq=" + ":".join(eq_params))
    if blur > 0.0: vf_filters.append(f"boxblur=luma_radius={blur:.3f}:luma_power=1")
    if sharp != 0.0: vf_filters.append(f"unsharp=lx=5:ly=5:la={sharp:.3f}")

    if use_fixed_res and 'x' in fixed_resolution:
        try:
            w, h = map(int, fixed_resolution.lower().split('x'))
            if w % 2 != 0 or h % 2 != 0:
                w = w - (w % 2); h = h - (h % 2)
                logger.warning(f"Adjusting resolution to {w}x{h} to meet even dimension requirements.")
            if rotation in ["90", "270"]:
                w, h = h, w
            
            if res_mode == "Crop": # Fill and crop
                vf_filters.append(f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}")
            elif res_mode == "Fit": # Letterbox
                vf_filters.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")
            else: # Stretch (default or "Stretch" explicitly)
                vf_filters.append(f"scale={w}:{h}")
        except ValueError: raise gr.Error(translations[language]["error_invalid_resolution_format"])
    elif scale_enabled and scale_factor != 100:
        new_width = int(frame_width * scale_factor / 100)
        new_height = int(frame_height * scale_factor / 100)
        vf_filters.append(f"scale={new_width}:{new_height}")

    cmd_final = [FFMPEG_PATH, "-framerate", str(input_ffmpeg_rate), "-i", str(frame_source)]
    tracks, vols = [], []
    if audio_track_1_tempfile:
        t = process_audio_track(1, audio_track_1_tempfile, fade_in_1, fade_out_1, final_video_duration, speed, preserve_pitch, sync_to_video=False, trim_to_video=trim_to_video, video_original_duration=final_video_duration)
        if t: tracks.append(t); vols.append(vol_track_1)
        else: logger.warning("Could not process audio track 1.")
    if audio_track_2_tempfile:
        t = process_audio_track(2, audio_track_2_tempfile, fade_in_2, fade_out_2, final_video_duration, speed, preserve_pitch, sync_to_video=False, trim_to_video=trim_to_video, video_original_duration=final_video_duration)
        if t: tracks.append(t); vols.append(vol_track_2)
        else: logger.warning("Could not process audio track 2.")
    if process_audio and is_conventional_video and vol_original > 0:
        speed_filter = f"atempo={speed:.4f}"
        if preserve_pitch and verify_audio_filters()[1]: speed_filter = f"rubberband=tempo={speed:.4f}"
        orig_audio = CACHE_DIR / "original_audio_processed.wav"
        try:
            subprocess.run([FFMPEG_PATH, "-i", str(input_path), "-vn", "-af", speed_filter, "-acodec", "pcm_s16le", "-y", str(orig_audio)], check=True, capture_output=True)
            if Path(orig_audio).exists():
                tracks.append(str(orig_audio)); vols.append(vol_original)
            else:
                logger.warning("Original audio processed file not generated.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not process original audio: {e.stderr}")
    
    output_dir = Path(output_path).parent
    if output_dir != Path(): output_dir.mkdir(parents=True, exist_ok=True)
    
    if not tracks:
        cmd_final.append("-an")
    else:
        for i, t in enumerate(tracks):
            if not Path(t).exists():
                raise gr.Error(f"{translations[language]['error_audio_track_not_found']} {t}")
            audio_duration = get_audio_duration(t)
            if audio_duration == 0:
                raise gr.Error(f"{translations[language]['error_audio_track_invalid_duration']} {t}")
            cmd_final.extend(["-i", t])
        if len(tracks) == 1:
            if vols[0] != 1.0:
                cmd_final.extend(["-filter_complex", f"[1:a]volume={vols[0]:.3f}[aout]"])
            else:
                cmd_final.extend(["-map", "0:v", "-map", "1:a"])
        else:
            mix_inputs = "".join(f"[{i+1}:a]volume={vol:.3f}[a{i+1}]" for i, vol in enumerate(vols) if vol != 1.0)
            if any(vol != 1.0 for vol in vols):
                mix_filter = f"{mix_inputs};{''.join(f'[a{i+1}]' for i in range(len(tracks)))}amix=inputs={len(tracks)}:duration=longest[aout]"
            else:
                mix_filter = f"{''.join(f'[{i+1}:a]' for i in range(len(tracks)))}amix=inputs={len(tracks)}:duration=longest[aout]"
            cmd_final.extend(["-filter_complex", mix_filter, "-map", "0:v", "-map", "[aout]"])
    
    if vf_filters:
        vf_filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
        cmd_final.extend(["-vf", ",".join(vf_filters)])
    else:
        cmd_final.extend(["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
    encoder = ["-c:v", "h264_nvenc", "-preset", "p5", "-qp", str(int(quality_crf))] if use_gpu and detect_gpu() else ["-c:v", "libx264", "-preset", "fast", "-crf", str(int(quality_crf))]
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=TEMP_DIR) as temp_output_file:
        temp_output_path = temp_output_file.name

    cmd_final.extend(encoder + ["-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-y", str(temp_output_path)])
    if use_shortest: cmd_final.append("-shortest")
    
    logger.info(f"Executing FFmpeg: {' '.join(map(str, cmd_final))}")
    try:
        # Ensure all command parts are strings for subprocess
        safe_cmd = [str(c) for c in cmd_final]
        result = subprocess.run(safe_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.info(f"Processing completed: {temp_output_path}")
    except subprocess.CalledProcessError as e:
        error_cmd = ' '.join(map(str, cmd_final))
        error_msg = f"{translations[language]['error_ffmpeg']} (code {e.returncode}):\n{e.stderr}\nCommand: {error_cmd}"
        logger.error(error_msg)
        raise gr.Error(error_msg)
    except Exception as e:
        error_cmd = ' '.join(map(str, cmd_final))
        error_msg = f"{translations[language]['error_video_processing_failed']}: {str(e)}\nCommand: {error_cmd}"
        logger.error(error_msg)
        raise gr.Error(error_msg)
    
    # --- Store result in In-Memory Cache ---
    output_info = get_video_info(temp_output_path)
    FINAL_VIDEO_CACHE[cache_key] = {
        "path": temp_output_path,
        "original_info": original_input_info,
        "output_info": output_info
    }
    logger.info(f"Stored result in cache with key {cache_key}")

    return temp_output_path, original_input_info, output_info

def clear_cache_and_temp():
    """Deletes all files and subdirectories in the cache and temp directories."""
    global FINAL_VIDEO_CACHE
    
    # Clear the in-memory cache
    FINAL_VIDEO_CACHE.clear()
    logger.info("In-memory cache cleared.")

    def clear_directory(directory):
        if directory.exists():
            # Check if it's a directory before iterating
            if directory.is_dir():
                for item in directory.iterdir():
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        logger.info(f"Removed {item}")
                    except Exception as e:
                        logger.error(f"Failed to delete {item}: {e}")

    logger.info("Starting cache and temp directory cleanup...")
    clear_directory(CACHE_DIR)
    clear_directory(TEMP_DIR)
    logger.info("Cache and temp directory cleanup finished.")
