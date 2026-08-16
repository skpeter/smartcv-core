from PyInstaller.utils.hooks import collect_data_files

# This file replaces the contrib hook-easyocr.py. Keep torch out of the
# freeze (AppData first-run) but still pack EasyOCR data files
# (character/*.txt, dict/*.txt, DBNet yaml). Without datas, Reader(['en'])
# dies on easyocr/character/en_char.txt.
excludedimports = ["torch", "torchvision", "torchaudio"]
hiddenimports = [
    "easyocr.model.vgg_model",
    "easyocr.model.model",
]
datas = collect_data_files("easyocr")
