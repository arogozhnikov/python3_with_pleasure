# Мысли о том, почему стоит мигрировать на питон3

- Питон 2 популярен среди ДС
- Однако питон 3 уже тоже весьма популярен, вот например Джейк https://jakevdp.github.io/blog/2013/01/03/will-scientists-ever-move-to-python-3/ писал про это и теперь уверен, что стоит двигаться в сторону питон 3
- Если вы только начинаете, то точно стоит сразу учить питон3
- Здесь гайд по переходу на третий питон, и какие плюшки это принесет


## Type hinting is now part of the language

```
def compute_time(data):
  data['time'] = data['distance'] / data['velocity'] 
```

which may work with dicy, pandas.DataFrame, astropy.Frame, numpy.recarray and a dozen of other containers.

Things are also quite complicated when operating with tensors, which may come of many different frameworks.

```
def convert_to_grayscale(images):
  return image.mean()
```


## Matrix multiplication as @

-

## Explicit difference between 'true division' and ''

-

## Strict ordering 

```python
3 < '3'
(3, 4) < (3, None)
```

- Prevents from occasional sorting of instances of different types
- 

## Main problems for code in data science 

