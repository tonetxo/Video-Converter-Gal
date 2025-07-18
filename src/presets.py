import json
import gradio as gr
from .config import PRESETS_FILE, translations

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

def save_preset(preset_name, *values, language="galego"):
    if not preset_name:
        gr.Warning(translations[language]["warning_no_preset_name"])
        return gr.update(choices=get_preset_choices())
    presets = load_presets_from_file()
    
    action_message = translations[language]['info_preset_saved']
    if preset_name in presets:
        action_message = translations[language]['info_preset_updated']

    keys = [
        "speed", "quality", "gpu", "scale_enabled", "scale_factor", "fixed_res_enabled", "fixed_res_value",
        "res_mode", "interpolate", "custom_fps", "fps_value", "rotation", "brightness", "contrast", "saturation", "sharp",
        "blur", "gamma", "process_audio", "preserve_pitch", "fade_in_1", "fade_out_1",
        "fade_in_2", "fade_out_2", "vol_original", "vol_track_1", "vol_track_2", "trim_to_video", "use_shortest"
    ]
    presets[preset_name] = dict(zip(keys, values))
    save_presets_to_file(presets)
    
    gr.Info(f"{action_message} '{preset_name}'!")
    
    return gr.update(choices=get_preset_choices(), value=preset_name)

def load_preset(preset_name, language="galego"):
    presets = load_presets_from_file()
    if preset_name in presets:
        gr.Info(f"{translations[language]['info_loading_preset']} '{preset_name}'...")
        p = presets[preset_name]
        values = [
            p.get("speed", 1.0), p.get("quality", 23), p.get("gpu", True),
            p.get("scale_enabled", False), p.get("scale_factor", 100),
            p.get("fixed_res_enabled", False), p.get("fixed_res_value", ""),
            p.get("res_mode", "Fit"), p.get("interpolate", False), p.get("custom_fps", False),
            p.get("fps_value", 25), p.get("rotation", "0"), p.get("brightness", 0.0),
            p.get("contrast", 1.0), p.get("saturation", 1.0),
            p.get("sharp", 0.0), p.get("blur", 0.0), p.get("gamma", 1.0),
            p.get("process_audio", True), p.get("preserve_pitch", True),
            p.get("fade_in_1", 1.0), p.get("fade_out_1", 1.0),
            p.get("fade_in_2", 1.0), p.get("fade_out_2", 1.0),
            p.get("vol_original", 1.0), p.get("vol_track_1", 1.0),
            p.get("vol_track_2", 1.0), p.get("trim_to_video", False),
            p.get("use_shortest", False)
        ]
        return values + [preset_name]
    gr.Warning(f"{translations[language]['warning_preset_not_found']} '{preset_name}'.")
    return [None] * 30

def delete_preset(preset_name, language="galego"):
    if not preset_name:
        gr.Warning(translations[language]["warning_no_preset_selected"])
        return gr.update(choices=get_preset_choices())
    presets = load_presets_from_file()
    if preset_name in presets:
        del presets[preset_name]
        save_presets_to_file(presets)
        gr.Info(f"{translations[language]['info_preset_deleted']} '{preset_name}'.")
        return gr.update(choices=get_preset_choices(), value=None)
    gr.Warning(f"{translations[language]['warning_preset_not_found']} '{preset_name}' {translations[language]['warning_to_delete']}.")
    return gr.update(choices=get_preset_choices())
