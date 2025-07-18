import gradio as gr
import os
from .config import translations, COLOR_PRESETS
from .presets import get_preset_choices, save_preset, load_preset, delete_preset
from .functions import (
    preview_color_correction_by_frame, apply_color_preset, process_video, create_preview,
    clear_cache_and_temp
)
from .utils import get_video_info, get_video_frame_count

def create_interface():
    def update_labels(language):
        t = translations[language]
        return (
            f"## 🎬 {t['title']}",
            t["input_file"], t["output_path"],
            t["preset_management"], t["load_preset"], t["load"], t["delete"],
            t["new_preset"], t["save"],
            t["main_options"], t["speed"], t["quality"], t["use_gpu"],
            t["resolution_fps"], t["scale_enabled"], t["scale_factor"],
            t["fixed_res_enabled"], t["fixed_res_value"], t["res_mode"],
            t["interpolate"], t["custom_fps"], t["fps_value"], t["rotation"],
            t["color_correction"], t["color_preset"], t["brightness"],
            t["contrast"], t["saturation"], t["sharp"], t["blur"], t["gamma"],
            t["preview_frame_label"],
            t["audio_options"], t["process_audio"], t["preserve_pitch"],
            t["trim_to_video"], t["use_shortest"], t["audio_track_1_label"],
            t["audio_track_1"], t["fade_in_1"], t["fade_out_1"],
            t["audio_track_2_label"], t["audio_track_2"], t["fade_in_2"],
            t["fade_out_2"], t["mixer"], t["vol_original"], t["vol_track_1"],
            t["vol_track_2"], t["process"], t["cancel"], t["quit"],
            t["result_video"], t["input_info"], t["output_info"],
            t["before"], t["after"], "Fotograma", t["clear_cache"]
        )

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        language = gr.Dropdown(choices=["galego", "english"], value="galego", label="Select Language")
        
        playable_preview_path = gr.State(None)
        original_video_info_state = gr.State("")

        with gr.Row():
            with gr.Column(scale=2):
                markdown = gr.Markdown(update_labels("galego")[0])
                input_file = gr.File(label=update_labels("galego")[1], file_types=['video', 'image'], file_count="single")
                output_path = gr.Textbox(label=update_labels("galego")[2], value="output.mp4", placeholder="Enter the output file name...")
                preset_acc = gr.Accordion(update_labels("galego")[3], open=True)
                with preset_acc:
                    preset_dropdown = gr.Dropdown(label=update_labels("galego")[4], choices=get_preset_choices(), interactive=True)
                    with gr.Row():
                        load_btn = gr.Button(translations["galego"]["load"])
                        delete_btn = gr.Button(translations["galego"]["delete"], variant="stop")
                    with gr.Row():
                        new_preset_name = gr.Textbox(label=update_labels("galego")[7], placeholder="Enter a name and press Save...")
                        save_btn = gr.Button(translations["galego"]["save"])
                main_opt_acc = gr.Accordion(update_labels("galego")[9], open=True)
                with main_opt_acc:
                    with gr.Row():
                        speed = gr.Slider(label=update_labels("galego")[10], minimum=0.25, maximum=4.0, value=1.0, step=0.05)
                        quality = gr.Slider(label=update_labels("galego")[11], minimum=14, maximum=35, value=23, step=1)
                    gpu = gr.Checkbox(label=update_labels("galego")[12], value=True)
                
                color_acc = gr.Accordion(update_labels("galego")[23], open=True)
                with color_acc:
                    color_preset = gr.Dropdown(label=update_labels("galego")[24], choices=list(COLOR_PRESETS.keys()), value="Normal")
                    with gr.Row():
                        brightness = gr.Slider(label=update_labels("galego")[25], minimum=-0.3, maximum=0.3, value=0.0, step=0.01)
                        contrast = gr.Slider(label=update_labels("galego")[26], minimum=0.7, maximum=1.5, value=1.0, step=0.01)
                        saturation = gr.Slider(label=update_labels("galego")[27], minimum=0.5, maximum=2.0, value=1.0, step=0.01)
                    with gr.Row():
                        sharp = gr.Slider(label=update_labels("galego")[28], minimum=-2.0, maximum=5.0, value=0.0, step=0.1)
                        blur = gr.Slider(label=update_labels("galego")[29], minimum=0.0, maximum=10.0, value=0.0, step=0.1)
                        gamma = gr.Slider(label=update_labels("galego")[30], minimum=0.1, maximum=10.0, value=1.0, step=0.1)
                    
                    frame_slider = gr.Slider(label="Fotograma", minimum=0, maximum=0, step=1, value=0, visible=False)
                    preview_btn = gr.Button(translations["galego"]["preview_frame_label"])

                res_fps_acc = gr.Accordion(update_labels("galego")[13], open=False)
                with res_fps_acc:
                    with gr.Row():
                        scale_enabled = gr.Checkbox(label=update_labels("galego")[14], value=False)
                        scale_factor = gr.Slider(label=update_labels("galego")[15], minimum=25, maximum=400, value=100, step=5)
                    fixed_res_enabled = gr.Checkbox(label=update_labels("galego")[16], value=False)
                    fixed_res_value = gr.Textbox(label=update_labels("galego")[17], placeholder="widthXheight")
                    res_mode = gr.Dropdown(label=update_labels("galego")[18], choices=["Crop", "Fit", "Fill"], value="Fit")
                    with gr.Row():
                        interpolate = gr.Checkbox(label=update_labels("galego")[19], value=False)
                        custom_fps = gr.Checkbox(label=update_labels("galego")[20], value=False)
                    fps_value = gr.Slider(label=update_labels("galego")[21], minimum=5, maximum=120, value=25.0, step=1)
                    rotation = gr.Dropdown(label=update_labels("galego")[22], choices=["0", "90", "180", "270"], value="0")

                audio_acc = gr.Accordion(update_labels("galego")[32], open=False)
                with audio_acc:
                    with gr.Row():
                        process_audio = gr.Checkbox(label=update_labels("galego")[33], value=True)
                        preserve_pitch = gr.Checkbox(label=update_labels("galego")[34], value=True)
                    trim_to_video = gr.Checkbox(label=update_labels("galego")[35], value=False)
                    use_shortest = gr.Checkbox(label=update_labels("galego")[36], value=False)
                    audio1_acc = gr.Accordion(update_labels("galego")[37], open=False)
                    with audio1_acc:
                        audio_track_1 = gr.File(label=update_labels("galego")[38], file_types=['audio'])
                        with gr.Row():
                            fade_in_1 = gr.Slider(label=update_labels("galego")[39], minimum=0.0, maximum=10.0, value=1.0, step=0.1)
                            fade_out_1 = gr.Slider(label=update_labels("galego")[40], minimum=0.0, maximum=10.0, value=1.0, step=0.1)
                    audio2_acc = gr.Accordion(update_labels("galego")[41], open=False)
                    with audio2_acc:
                        audio_track_2 = gr.File(label=update_labels("galego")[42], file_types=['audio'])
                        with gr.Row():
                            fade_in_2 = gr.Slider(label=update_labels("galego")[43], minimum=0.0, maximum=10.0, value=1.0, step=0.1)
                            fade_out_2 = gr.Slider(label=update_labels("galego")[44], minimum=0.0, maximum=10.0, value=1.0, step=0.1)
                    mixer_acc = gr.Accordion(update_labels("galego")[45], open=False)
                    with mixer_acc:
                        vol_original = gr.Slider(label=update_labels("galego")[46], minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                        vol_track_1 = gr.Slider(label=update_labels("galego")[47], minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                        vol_track_2 = gr.Slider(label=update_labels("galego")[48], minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                
                process_btn = gr.Button(translations["galego"]["process"], variant="primary")
                cancel_btn = gr.Button(translations["galego"]["cancel"])
                clear_cache_btn = gr.Button(translations["galego"]["clear_cache"])
                quit_btn = gr.Button(translations["galego"]["quit"])
            
            with gr.Column(scale=3):
                preview_video = gr.Video(label="Previsualización", interactive=False, height=270)
                input_info = gr.Markdown(label=update_labels("galego")[53])
                with gr.Row(visible=True) as preview_row:
                    before_preview = gr.Image(label=update_labels("galego")[55])
                    after_preview = gr.Image(label=update_labels("galego")[56])
                result_video = gr.Video(label=update_labels("galego")[52], interactive=False, height=270)
                output_info = gr.Markdown(label=update_labels("galego")[54])

        def update_preview_and_info(input_file_obj):
            if not input_file_obj:
                return None, "No file uploaded.", None, gr.update(visible=False), ""
            
            preview_path, original_file_info = create_preview(input_file_obj)
            if not preview_path:
                return None, "Could not process file.", None, gr.update(visible=False), ""

            frame_count = get_video_frame_count(preview_path)
            
            slider_update = gr.update(visible=True, maximum=frame_count - 1 if frame_count > 0 else 0, value=0)
            
            return preview_path, original_file_info, preview_path, slider_update, original_file_info

        def update_all_labels(language):
            labels = update_labels(language)
            return (
                gr.Markdown(labels[0]), gr.File(label=labels[1]), gr.Textbox(label=labels[2]),
                gr.Accordion(labels[3]), gr.Dropdown(label=labels[4]), gr.Button(labels[5]),
                gr.Button(labels[6]), gr.Textbox(label=labels[7]), gr.Button(labels[8]),
                gr.Accordion(labels[9]), gr.Slider(label=labels[10]), gr.Slider(label=labels[11]),
                gr.Checkbox(label=labels[12]), gr.Accordion(labels[13]), gr.Checkbox(label=labels[14]),
                gr.Slider(label=labels[15]), gr.Checkbox(label=labels[16]), gr.Textbox(label=labels[17]),
                gr.Dropdown(label=labels[18]), gr.Checkbox(label=labels[19]), gr.Checkbox(label=labels[20]),
                gr.Slider(label=labels[21]), gr.Dropdown(label=labels[22]), gr.Accordion(labels[23]),
                gr.Dropdown(label=labels[24]), gr.Slider(label=labels[25]), gr.Slider(label=labels[26]),
                gr.Slider(label=labels[27]), gr.Slider(label=labels[28]), gr.Slider(label=labels[29]),
                gr.Slider(label=labels[30]), gr.Button(labels[31]), gr.Accordion(labels[32]),
                gr.Checkbox(label=labels[33]), gr.Checkbox(label=labels[34]), gr.Checkbox(label=labels[35]),
                gr.Checkbox(label=labels[36]), gr.Accordion(labels[37]), gr.File(label=labels[38]),
                gr.Slider(label=labels[39]), gr.Slider(label=labels[40]), gr.Accordion(labels[41]),
                gr.File(label=labels[42]), gr.Slider(label=labels[43]), gr.Slider(label=labels[44]),
                gr.Accordion(labels[45]), gr.Slider(label=labels[46]), gr.Slider(label=labels[47]),
                gr.Slider(label=labels[48]), gr.Button(labels[49]), gr.Button(labels[50]),
                gr.Button(labels[51]), gr.Video(label=labels[52]), gr.Markdown(label=labels[53]),
                gr.Markdown(label=labels[54]), gr.Image(label=labels[55]), gr.Image(label=labels[56]),
                gr.Slider(label=labels[57]), gr.Button(labels[58])
            )

        preset_components = [
            speed, quality, gpu, scale_enabled, scale_factor, fixed_res_enabled,
            fixed_res_value, res_mode, interpolate, custom_fps, fps_value, rotation, brightness,
            contrast, saturation, sharp, blur, gamma, process_audio, preserve_pitch,
            fade_in_1, fade_out_1, fade_in_2, fade_out_2, vol_original,
            vol_track_1, vol_track_2, trim_to_video, use_shortest
        ]
        
        color_correction_components = [brightness, contrast, saturation, sharp, blur, gamma]

        all_process_inputs = [
            input_file, output_path, original_video_info_state, speed, quality, gpu, scale_enabled, scale_factor,
            custom_fps, fps_value, interpolate, fixed_res_enabled, fixed_res_value,
            res_mode, rotation, process_audio, preserve_pitch, audio_track_1, fade_in_1, fade_out_1,
            audio_track_2, fade_in_2, fade_out_2, vol_original, vol_track_1,
            vol_track_2, brightness, contrast, saturation, sharp, blur, gamma, trim_to_video, use_shortest
        ]
        
        def handle_clear_cache(lang):
            clear_cache_and_temp()
            gr.Info(translations[lang]["cache_cleared_message"])

        input_file.change(
            fn=update_preview_and_info, 
            inputs=input_file, 
            outputs=[preview_video, input_info, playable_preview_path, frame_slider, original_video_info_state]
        )
        frame_slider.change(
            fn=preview_color_correction_by_frame,
            inputs=[playable_preview_path, frame_slider] + color_correction_components + [language],
            outputs=[before_preview, after_preview]
        )
        preview_btn.click(
            fn=preview_color_correction_by_frame,
            inputs=[playable_preview_path, frame_slider] + color_correction_components + [language],
            outputs=[before_preview, after_preview]
        )
        color_preset.change(fn=apply_color_preset, inputs=[color_preset, language], outputs=color_correction_components)
        process_event = process_btn.click(fn=process_video, inputs=all_process_inputs, outputs=[result_video, input_info, output_info])
        cancel_btn.click(fn=None, inputs=None, outputs=None, cancels=[process_event])
        save_btn.click(fn=save_preset, inputs=[new_preset_name] + preset_components + [language], outputs=preset_dropdown)
        load_btn.click(fn=load_preset, inputs=[preset_dropdown, language], outputs=preset_components + [new_preset_name])
        delete_btn.click(fn=delete_preset, inputs=[preset_dropdown, language], outputs=preset_dropdown)
        clear_cache_btn.click(fn=handle_clear_cache, inputs=[language], outputs=None)
        
        preset_dropdown.change(fn=lambda x: x, inputs=preset_dropdown, outputs=new_preset_name)
        demo.load(fn=lambda x: x, inputs=preset_dropdown, outputs=new_preset_name)
        
        language.change(
            fn=update_all_labels,
            inputs=language,
            outputs=[
                markdown, input_file, output_path, preset_acc, preset_dropdown, load_btn, delete_btn,
                new_preset_name, save_btn, main_opt_acc, speed, quality, gpu, res_fps_acc,
                scale_enabled, scale_factor, fixed_res_enabled, fixed_res_value, res_mode,
                interpolate, custom_fps, fps_value, rotation, color_acc, color_preset,
                brightness, contrast, saturation, sharp, blur, gamma, preview_btn, audio_acc,
                process_audio, preserve_pitch, trim_to_video, use_shortest, audio1_acc, audio_track_1,
                fade_in_1, fade_out_1, audio2_acc, audio_track_2, fade_in_2, fade_out_2, mixer_acc,
                vol_original, vol_track_1, vol_track_2, process_btn, cancel_btn, quit_btn,
                result_video, input_info, output_info, before_preview, after_preview, frame_slider,
                clear_cache_btn
            ]
        )
        
        def shutdown_app():
            os._exit(0)
        quit_btn.click(fn=shutdown_app)
    
    return demo
