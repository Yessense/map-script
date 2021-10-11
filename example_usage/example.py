from src.mapscript.preprocessing.extract_texts_info import extract_texts_info
from src.mapscript.script import Script
from src.mapscript.visualization.visualizator import Visualizator
from src.mapscript.vsa import encoder

filepath = '/home/yessense/PycharmProjects/ScriptExtractionForVQA/Approved/jack/restaurant.txt'
text_info = extract_texts_info([filepath])[0]
script = Script(text_info)
script_encoder = encoder.ScriptEncoder(script, 1)
sim = script_encoder.check_vectors('mill', 1)

vis = Visualizator(script, save_to_file=False, show_buttons=False)

vis.show()
print("Done")
