from diffusers import AudioLDMPipeline
import torch
import os
import folder_paths
from loguru import logger
from huggingface_hub import snapshot_download
from modelscope import snapshot_download as ms_snapshot_download
import random
import json
import scipy

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="safetensors")

_model = None

MODELSCOPE_HUGGINGFACE_MAP = {
    "cvssp/audioldm-l-full": "cutemodel/audioldm-l-full",
}

def check_huggingface_access() -> bool:
    url = "https://huggingface.co"
    try:
        import requests
        _ = requests.get(url, timeout=5)
        return True
    except:
        return False

def download_model_hf(model_path: str, repo_id: str) -> bool:
    hf_access_result = check_huggingface_access()
    if not hf_access_result:
        logger.error("download model failed, try to use mirror source")
        endpoint = "https://hf-mirror.com"
    else:
        endpoint = None

    logger.info(f"download model from {repo_id} to {model_path}")
    try:
        logger.info(f"use huggingface mirror source to download model, if failed, try to download model from modelscope")
        snapshot_download(repo_id=repo_id,
                        local_dir=model_path,
                        local_dir_use_symlinks=False,
                        endpoint=endpoint)
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

def download_model_modelscope(model_path: str, model_id: str) -> bool:
    logger.info(f"download model from ModelScope {model_id} to {model_path}")
    try:
        ms_snapshot_download(model_id, cache_dir=model_path, revision='master')
        return True
    except Exception as e:
        logger.error(f"Failed to download model from ModelScope: {e}")
        return False

class AudioLDM:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "prompt": ("STRING", {"multiline": True}),
                        "audio_length": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                        "num_steps": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                        "sample_rate": ("INT", {"default": 16000, "min": 8000, "max": 48000, "step": 100}),
                        "seed": ("INT", {"default": 123456, "min": 0, "max": 0xffffffffffffffff}),
                    }
                }

    CATEGORY = "audio"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_audio"
    def generate_audio(self, prompt: str, audio_length: float, num_steps: int, sample_rate: int, seed: int):
        # 此处 seed 无实际用途，只是为了让 ComfyUI 支持刷新，否则固定参数只能生成一个结果
        global _model

        if _model is None:
            model_path = os.path.join(folder_paths.models_dir, "audioldm-l-full")
            
            if not check_huggingface_access():
                model_id = MODELSCOPE_HUGGINGFACE_MAP["cvssp/audioldm-l-full"]
                download_model_modelscope(model_path, model_id)
                model_path = os.path.join(model_path, model_id.split("/")[0], model_id.split("/")[1])
            else:
                model_id = "cvssp/audioldm-l-full"
                download_model_hf(model_path, model_id)
                model_path = model_path

            _model = AudioLDMPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
            _model = _model.to("cuda")

        # 设置随机种子  
        # generator = torch.Generator("cuda").manual_seed(seed)   

        audio = _model(
            prompt,
            num_inference_steps=num_steps,
            audio_length_in_s=audio_length,
        ).audios[0]

        audio_dict = {
            "waveform": audio,
            "sample_rate": sample_rate
        }
        return (audio_dict,)
class SaveAudioLDM:
    def __init__(self):
        self.comfyui_output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                              "filename_prefix": ("STRING", {"default": "test_audio"}),
                              "output_folder_name": ("STRING", {"default": "audio"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    CATEGORY = "audio"

    def save_audio(self, audio, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None, output_folder_name="audio"):
        output_dir = os.path.join(self.comfyui_output_dir, output_folder_name)
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
        results = list()

        metadata = {}
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

        filename_with_batch_num = filename.replace("%batch_num%", str(0))
        file = f"{filename_with_batch_num}_{counter:05}_.flac"
        scipy.io.wavfile.write(os.path.join(full_output_folder,file), rate=audio["sample_rate"], data=audio["waveform"])

        print(subfolder,file,self.type)
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        counter += 1

        return { "ui": { "audio": results } }

class PreviewAudioLDM(SaveAudioLDM):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIO", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

NODE_CLASS_MAPPINGS = {
    "AudioLDM": AudioLDM,
    "SaveAudioLDM": SaveAudioLDM,
    "PreviewAudioLDM": PreviewAudioLDM,
}