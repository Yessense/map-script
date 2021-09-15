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
files = [filepath]
text_info = extract_texts_info(files)[0]

script = Script(text_info)

# script visualization
vis = Visualizator(script, save_to_file=False)
vis.show()
```

[comment]: <> (## Text analysis )
[comment]: <> (```text)

[comment]: <> (A person went into a cafe.)

[comment]: <> (He sat down at a table and then ordered the dish of the day.)

[comment]: <> (The waiter brought him a hot meal.)

[comment]: <> (The man slowly finished the dish and then asked for the bill.)

[comment]: <> (He paid and left the cafe.)

[comment]: <> (```)

## Visualization
### Text

![Example](./img/separate_scripts_example.jpg?raw=true "Example")

#### Sentence 1

![Sentence 1.](./img/text_1.png?raw=true "Sentence 1")

#### Sentence 1 - 2 
![Sentence 1 - 2.](./img/text_2.png?raw=true "Sentence 2")

#### Sentence 1 - 3
![Sentence 1 - 3.](./img/text_3.png?raw=true "Sentence 3")

#### Sentence 1 - 4
![Sentence 1 - 4.](./img/text_4.png?raw=true "Sentence 4")

#### Sentence 1 - 5
![Sentence 1 - 5.](./img/text_5.png?raw=true "Sentence 5")

#### Sentence 1 - 6
![Sentence 1 - 6.](./img/text_6.png?raw=true "Sentence 6")
