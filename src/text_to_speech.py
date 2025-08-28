# src/text_to_speech.py

import subprocess
import os
from pathlib import Path

# --- ИСПРАВЛЕНИЕ: Мы теперь в папке src, поэтому нужно подняться на один уровень выше ---
PROJECT_ROOT = Path(__file__).parent.parent 
PIPER_DIR = PROJECT_ROOT / "piper"
PIPER_EXE_PATH = PIPER_DIR / "piper.exe"
MODELS_DIR_RU = PIPER_DIR / "ru"
MODELS_DIR_EN = PIPER_DIR / "en"

class Text_to_speech:
    def __init__(self):
        if not PIPER_EXE_PATH.exists():
            raise FileNotFoundError(f"Piper executable not found at: {PIPER_EXE_PATH}")

        self.md_ru = self._load_models(MODELS_DIR_RU)
        self.md_en = self._load_models(MODELS_DIR_EN)

    def _load_models(self, models_path: Path) -> dict:
        """Сканирует указанную папку и загружает пути к моделям .onnx."""
        model_dict = {}
        if not models_path.exists():
            print(f"Warning: Models directory not found: {models_path}")
            return {}

        for onnx_file in models_path.rglob("*.onnx"):
            try:
                parts = onnx_file.stem.split('-')
                model_name = parts[1]
                quality = parts[2]
                
                if model_name not in model_dict:
                    model_dict[model_name] = []
                model_dict[model_name].append({'quality': quality, 'path': onnx_file})

            except IndexError:
                print(f"Warning: Could not parse model name from file: {onnx_file.name}")
                continue
        
        quality_order = {'high': 0, 'medium': 1, 'low': 2}
        for name, models in model_dict.items():
            models.sort(key=lambda m: quality_order.get(m['quality'], 99))
            
        return model_dict

    def speak(self, text: str, model: str, lang: str, file_name: str, output_path: str):
        """Генерирует аудиофайл из текста с помощью указанной модели."""
        model_collection = self.md_ru if lang == "ru" else self.md_en
        
        if model not in model_collection or not model_collection[model]:
            print(f"Warning: Model '{model}' not found for language '{lang}'. Skipping TTS.")
            return

        model_path = model_collection[model][0]['path']
        output_wav_path = Path(output_path) / f"{file_name}.wav"

        command = [
            str(PIPER_EXE_PATH),
            '--model', str(model_path),
            '--output_file', str(output_wav_path),
        ]
        
        try:
            subprocess.run(
                command, 
                input=text.encode('utf-8'), 
                check=True, 
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running Piper TTS: {e}")
            print(f"Stderr: {e.stderr.decode('utf-8', errors='ignore')}")
        except FileNotFoundError:
            print(f"Error: Piper executable not found at {PIPER_EXE_PATH}")