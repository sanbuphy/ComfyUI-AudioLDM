import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.audioldm_l import AudioLDM, SaveAudioLDM, PreviewAudioLDM

NODE_CLASS_MAPPINGS = {
    "AudioLDM": AudioLDM,
    "SaveAudioLDM": SaveAudioLDM, 
    "PreviewAudioLDM": PreviewAudioLDM,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioLDM": "AudioLDM",
    "SaveAudioLDM": "SaveAudioLDM",
    "PreviewAudioLDM": "PreviewAudioLDM",
}

all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']