import hashlib
import os
import pickle

from src.script_extraction.sign.extract_script import create_signs
from src.script_extraction.visualization.show_script import show_script

saved_files_dir = '/texts/saved/'
saved_files = [os.path.basename(f) for f in os.listdir(saved_files_dir)]
TEXT_FOLDER = "/home/yessense/PycharmProjects/ScriptExtractionForVQA/texts/Race/"

nice_texts = ['80.txt', '59.txt', '43.txt', '84.txt', '24.txt']
files = [os.path.join(TEXT_FOLDER, f)
         for f in os.listdir(TEXT_FOLDER)
         if os.path.isfile(os.path.join(TEXT_FOLDER, f))]

for filepath in files:
    h = hashlib.sha1(filepath.encode()).hexdigest()
    if h in saved_files:
        with open(os.path.join(saved_files_dir, h), 'rb') as f:
            info = pickle.load(f)
            script = create_signs(info)
            show_script(script, group_roles=True)
            print(f'{filepath} processed')

