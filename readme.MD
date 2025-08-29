


```python
import os

for i in  range(10):
    dir_text = !dir
    if any([".venv" in text for text in dir_text]):
        print(".venv encontrado")
        break
    else:
        print(f'{i} - os.chdir("..")')
        os.chdir("..")

    
%load_ext autoreload
%autoreload 2
```