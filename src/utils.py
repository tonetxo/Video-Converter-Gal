import gradio as gr
import subprocess
import json
from PIL import Image
import logging
from pathlib import Path
from .config import FFMPEG_PATH, FFPROBE_PATH, CACHE_DIR, TEMP_DIR

logger = logging.getLogger(__name__)

def verify_tools():
    if not all([FFMPEG_PATH, FFPROBE_PATH]):
        raise gr.Error("FFmpeg and FFprobe are required.")
    CACHE_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    logger.info("Tools verified correctly")

def get_video_duration(file_path):
    try:
        cmd = [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        return float(result.stdout.strip())
    except:
        try:
            with Image.open(file_path) as img:
                n_frames = getattr(img, 'n_frames', 1)
                duration_ms = img.info.get('duration', 40)
                return (n_frames * duration_ms) / 1000.0 if n_frames > 1 else 0.0
        except: return 0.0

def get_audio_duration(file_path):
    try:
        cmd = [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        return float(result.stdout.strip())
    except: return 0.0

def get_video_info(file_path):
    if not file_path or not Path(file_path).exists(): return "The file does not exist."
    try:
        cmd = [FFPROBE_PATH, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(file_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        data = json.loads(result.stdout)
        video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
        if video_stream:
            info = [f"**Format:** {data['format'].get('format_long_name', 'N/A')}"]
            fps_str = video_stream.get('r_frame_rate', '0/1')
            try: num, den = map(int, fps_str.split('/')); fps = f"{num / den:.2f}" if den > 0 else "0"
            except: fps = "N/A"
            info.extend([f"**Resolution:** {video_stream.get('width', 'N/A')}x{video_stream.get('height', 'N/A')}", f"**FPS:** {fps}", f"**Video Codec:** {video_stream.get('codec_name', 'N/A')}"])
            frames = video_stream.get('nb_frames', 'N/A')
            if frames != 'N/A': info.append(f"**Number of Frames:** {frames}")
            audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
            if audio_stream:
                info.append(f"**Audio Codec:** {audio_stream.get('codec_name', 'N/A')}")
                sample_rate = audio_stream.get('sample_rate', 'N/A')
                if sample_rate != 'N/A': info.append(f"**Sample Rate:** {sample_rate} Hz")
            else: info.append("**Audio:** No audio track")
            duration = float(data['format'].get('duration', 0))
            info.append(f"**Duration:** {duration:.2f} seconds")
            bitrate = data['format'].get('bit_rate', 'N/A')
            if bitrate != 'N/A': info.append(f"**Bitrate:** {int(bitrate) / 1_000_000:.2f} Mbps")
            info_string = "\n".join(info)
            return info_string
    except Exception as e:
        logger.debug(f"FFprobe failed for {file_path}: {e}") # Log FFprobe failure

    # Fallback for image formats (especially animated ones)
    try:
        with Image.open(file_path) as img:
            w, h = img.size
            n_frames = getattr(img, 'n_frames', 1)
            duration_ms = img.info.get('duration', 40) # Default to 40ms per frame if not specified
            
            # Calculate total duration in seconds
            total_s = (n_frames * duration_ms) / 1000.0 if n_frames > 1 else 0.0
            
            # Calculate FPS
            fps = n_frames / total_s if total_s > 0 else 25.0 # Default to 25 FPS if duration is 0

            info = [
                f"**Format:** Animated Image ({img.format})",
                f"**Resolution:** {w}x{h}",
                f"**FPS:** {fps:.2f}",
                f"**Video Codec:** {img.format.lower()}", # Use image format as video codec
                f"**Number of Frames:** {n_frames}",
                "**Audio:** No audio track", # Animated images typically have no audio
                f"**Duration:** {total_s:.2f} seconds",
                "**Bitrate:** N/A" # Bitrate is not directly applicable to image sequences in this context
            ]
            info_string = "\n".join(info)
            return info_string
    except Exception as e:
        logger.error(f"Could not read the file with Pillow: {e}", exc_info=True)
        return f"Could not read the file.\nError: {e}"

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
                duration_ms = img.info.get('duration', 40)
                if n_frames <= 1 or duration_ms <= 0: return 25.0
                return n_frames / ((n_frames * duration_ms) / 1000.0) if (n_frames * duration_ms) / 1000.0 > 0 else 25.0
        except: return 25.0

def get_frame_dimensions(frame_path):
    try:
        with Image.open(frame_path) as img:
            return img.width, img.height
    except Exception as e:
        logger.error(f"Could not get frame dimensions {frame_path}: {e}")
        return 0, 0

def get_video_frame_count(file_path):
    if not file_path or not Path(file_path).exists():
        return 0
    try:
        cmd = [
            FFPROBE_PATH,
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        # Fallback for image formats
        try:
            with Image.open(file_path) as img:
                return getattr(img, 'n_frames', 1)
        except Exception:
            return 0
    except Exception:
        return 0
