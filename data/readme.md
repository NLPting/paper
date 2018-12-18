# How to use

## json 格式  
- class Word  
    - str word  
    - POS list poses  
- class POS   
    - str pos  
    - Sense list senses  
- class Sense  
    - str sense  
    - str level  
    - str dict_examp  
    - str lear_examp

## Code

```python
json_data = json.load(open('A.json'))
for word in json_data:
    head = word['word']
    for poses in word['poses']:
        pos = poses['pos']
        for senses in poses['senses']:
            dict_examp = senses['dict_examp']
            lear_examp = senses['lear_examp']
            sense = senses['sense']
```

