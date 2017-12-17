# Мысли о том, почему стоит мигрировать на питон3

- Питон 2 популярен среди ДС
- Однако питон 3 уже тоже весьма популярен, вот например Джейк https://jakevdp.github.io/blog/2013/01/03/will-scientists-ever-move-to-python-3/ писал про это и он теперь уверен, что стоит двигаться в сторону питон 3
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

Еще нужно return doctypes продемонстрировать, IDE контролирует, когда возвращается что-то не то.

def train_on_batch(batch_data: tensor, batch_labels: tensor) -> Tuple[tensor, float]:
  ...
  return loss, accuracy

(In most cases) IDE will spot an error if you forgot to convert an accuracy to float.

## Matrix multiplication as @

-

## Print Is A Function

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

-

## Unicode 

Значительно проще работать с текстами, если вы занимаетесь NLP. 

TODO надо ссылочку на хорошие примеры.

## Default pickle engine provides much better compression


## Single integer type

```
isinstance()


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


