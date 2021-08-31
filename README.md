# map-script

Package for extracting scripts from English texts

## Package Installation

Do the following steps:

1. Install 'map-core' from [the repository](https://github.com/cog-isa/map-core) 
   > `pip install git+https://github.com/cog-isa/map-core.git`
3. Install package
   >`pip install git+https://github.com/Yessense/map-script.git`

## Example usage


```python
from mapscript.preprocessing.extract_texts_info import extract_texts_info
from mapscript.script import Script
from mapscript.visualization.visualizator import Visualizator

filepath = 'my_text.txt'
text_info = extract_texts_info([filepath])[0]

script = Script(text_info)

# script visualization
vis = Visualizator(script, save_to_file=False)
vis.show()
```
