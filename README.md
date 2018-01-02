# Guide on migrating to python3 for data scientists

This guide should help you to move to python 3 and enjoy it! 
Most probably, you already know about the problems caused by inconsistencies between python2 and python3, here I cover some of the changes that may come in handy for data scientists.

- Питон 2 популярен среди ДС
- Однако питон 3 уже тоже весьма популярен, вот например Джейк https://jakevdp.github.io/blog/2013/01/03/will-scientists-ever-move-to-python-3/ писал про это и он теперь уверен, что стоит двигаться в сторону питон 3
- Если вы только начинаете, то точно стоит сразу учить питон3
- Здесь гайд по переходу на третий питон, и какие плюшки это принесет
- Мне не нравится учить людей питону3, потому что требуется сразу объяснять iterables, которые не слишком нужны, но путают


## Type hinting is now part of the language

```
def compute_time(data):
  data['time'] = data['distance'] / data['velocity'] 
```

which may work with dict, pandas.DataFrame, astropy.Frame, numpy.recarray and a dozen of other containers.

Things are also quite complicated when operating with tensors, which may come of many different frameworks.

```
def convert_to_grayscale(images):
  return image.mean()
```

Еще нужно return doctypes продемонстрировать, IDE контролирует, когда возвращается что-то не то.

def train_on_batch(batch_data: tensor, batch_labels: tensor) -> Tuple[tensor, float]:
  ...
  return loss, accuracy

(In most cases) IDE will spot an error if you forgot to convert an accuracy to float.

## Matrix multiplication as @

-

## Better paths handling

`pathlib` is a default module in python3. 
Use it to avoid tons of `os.path.join`s:

```python
from pathlib import Path
dataset_root = Path('/path/to/dataset/')
train_path = dataset_root / 'train'
test_path = dataset_root / 'test'
for image_path in train_path.iterdir():
    with image_path.open() as f: # note, open is a method of 
        # do something with an image
```

It's always tempting to use string concatenation (which is obviously bad, but more verbose), 
in python3 code is both safe and verbose.

Also `pathlib.Path` has a bunch of methods, that every python novice has to google (and anyone who is not working with files all the time):
```python
p.exists()
p.is_dir()
p.parts()
p.withsuffix('.jpg') # only change the extension!
p.chmod(mode)
```


## Globbing with `**`


## Print Is A Function Now

Of course, you have already learnet this, but apart from adding annoying parenthesis, there are some advantages:

-

## Explicit difference between 'true division' and 'integer division'

While this change may be not a good fit for system programming in python, for data science this is definitely a positive change

```python
velocity = distance / time

data = pandas.read_csv('timing.csv')
velocity = data['distance'] / data['time']
```

Result in python2 depends on whether 'time' and 'distance' (e.g. measured in meters and seconds) are stored integer.
In python3, result is correct in both cases, promotion. 

Another case is integer division, which is now an eplicit operation:

```python
n_gifts = money // gift_price
```

## Strict ordering 

```python
3 < '3'
2 < None
(3, 4) < (3, None)
(4, 5) < [4, 5]
(4, 5) == [4, 5]
```

- Prevents from occasional sorting of instances of different types
- Generally, resolves some problems that arise when processing raw data

Sidenote: proper check for None is
```python
if a is not None:
  pass
  
if a: # very bad idea!
  pass
```

## Iterable unpacking

```python
# handy when amound of additional stored info may vary between experiments, but the same code can be used in all cases
model_paramteres, optimizer_parameters, *other_params = load(checkpoint_name)

# picking two last values from a sequence
*prev, next_to_last, last = values_history

# This also works with any iterables, so if you have a function that yields, say, qualities
# simple way to take only last two values from a list
*prev, next_to_last, last = iter_train(args)
```

## OrderedDict is faster now

OrderedDict is probably the most used structure after list. It a good old dictionary, which keeps the order in which keys were added.
-

## Unicode 

Значительно проще работать с текстами, если вы занимаетесь NLP. 

TODO надо ссылочку на хорошие примеры.

## Default pickle engine provides better compression for arrays


## Single integer type

Python2 provides two basic integer types, which are `int` (64-bit signed integer) and `long` (for long arithmetics).

Python3 now has only `int`, which provides long arithmetics.

Checking for integer is easier in python 3:

```
isinstance(x, numbers.Integral) # python2
isinstance(x, [long, int]) # python2
isinstance(x, int) # python3, easiest to remember
```

# 

## Other 

- yield from 
- async / await


## Main problems for code in data science and how to resolve those

- relative imports from subfloders
  - packaging 
  - sys.path.insert
  - softlinks
  
- support for nested arguments was dropped
  ```
  map(lambda x, (y, z): x, z, dict.items())
  ```
  
  However, it is still perfectly working with different comprehensions:
  ```python
  {x:z for x, (y, z) in d.items()}
  ```
  In general, comprehensions are also much better 'translatable' between python2 and python 3.

- map, values, items do not return lists.
  Problem with enumerators are:
  - no trivial slicing
  - no double usage
  
  Quite typically, all of this situations are resolved by converting to list.


