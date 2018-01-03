# Moving to python 3 with pleasure
## A short guide on migrating from python2 to python3 for data scientists

Most probably, you already know about the problems caused by inconsistencies between python2 and python3, 
here I cover some of the changes that may come in handy for data scientists.

- Питон 2 популярен среди ДС
- Однако питон 3 уже тоже весьма популярен, вот например Джейк https://jakevdp.github.io/blog/2013/01/03/will-scientists-ever-move-to-python-3/ писал про это и он теперь уверен, что стоит двигаться в сторону питон 3
- Если вы только начинаете, то точно стоит сразу учить питон3
- Здесь гайд по переходу на третий питон, и какие плюшки это принесет
- Мне не нравится учить людей питону3, потому что требуется сразу объяснять iterables, которые не слишком нужны, но путают



## Better paths handling

`pathlib` is a default module in python3, that helps you to avoid tons of `os.path.join`s:

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
in python3 code is safe, concise, and readable.

Also `pathlib.Path` has a bunch of methods, that every python novice has to google (and anyone who is not working with files all the time):

```python
p.exists()
p.is_dir()
p.parts() 
p.with_name('sibling.png') # only change the name, but keep the folder
p.with_suffix('.jpg') # only change the extension, but keep the folder and the name
p.chmod(mode)
p.rmdir()
```

`Pathlib` should save you lots of time, 
please see [docs](https://docs.python.org/3/library/pathlib.html) and [reference](https://pymotw.com/3/pathlib/) for more.


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


## Type hinting for conversion

TODO complete this one

## Matrix multiplication as @

Let's implement one of the simplest ML models - a linear regression with l2 regularization:

```
# L2-regularized linear regression: || AX - b ||^2 + alpha * ||x||^2 -> min

# python2
X = np.linalg.inv(np.dot(A.T, A) + alpha * np.eye(A.shape[1])).dot(A.T.dot(b))
# python3
X = np.linalg.inv(A.T @ A + alpha * np.eye(A.shape[1])) @ (A.T @ b)
```

The code with `@` becomes more readable and more translatable between deep learning frameworks: same code `X @ W + b[None, :]` for a single layer of perceptron works in `numpy`, `cupy`, `pytorch`, `tensorflow` (and other frameworks that operate with tensors).


## Globbing with `**`

Recursive folder globbing is not easy in python2, even custom module [glob2](https://github.com/miracle2k/python-glob2) was written to overcome this. Since python3.6 recurive flag is supported:

```python
import glob
# python2

found_images = \
    glob.glob('/path/*/*.jpg') \
  + glob.glob('/path/*/*/*.jpg') \
  + glob.glob('/path/*/*/*/*.jpg') \
  + glob.glob('/path/*/*/*/*/*.jpg') \
  + glob.glob('/path/*/*/*/*/*/*.jpg') 

# python3

found_images = glob.glob('/path/**/*.jpg', recursive=True)
```

Another option is to use `pathlib` in python3 (minus one import!):
```python
found_images = pathlib.Path('/path').glob('**/*.jpg')
```

## Print Is A Function Now

Probably, you've already learnt this, but apart from adding annoying parenthesis, there are some advantages:

You don't need to remember the special syntax for using file descriptor:
```python
print >>sys.stderr, "fatal error" # python2
print("fatal error", file=sys.stderr) # python3
```

Finally, parentheses are not as annoying after a couple of months :)

It also worth mentioning, that printing of tab-aligned tables can be done without `str.join`:
```python
print(*array, sep='\t')
print(batch, epoch, loss, accuracy, time, sep='\t')
```
## Formatted string literals for logging

Quite typically data scientist outputs iteratively some logging information as in a fixed format. 

```python
print('{batch:3} {epoch:3} / {total_epochs:3}  accuracy: {acc_mean:0.4f}±{acc_std:0.4f} time: {avg_time:3.2f}'.format(
    batch=batch, epoch=epoch, total_epochs=total_epochs, 
    acc_mean=numpy.mean(accuracies), acc_std=numpy.std(accuracies),
    avg_time=time / len(data_batch)
))


print(f'{batch:3} {epoch:3} / {total_epochs:3}  accuracy: {numpy.mean(accuracies):0.4f}±{numpy.std(accuracies):0.4f} time: {time / len(data_batch):3.2f}')
```

Sample output:
```
120  12 / 300  accuracy: 0.8180±0.4649 time: 56.60
```

Default logging system provides the flexibility (template and formatted values are independent) that is not required in research code. 
This comes at the cost of being either too verbose and writing the code that is too prone to errors during editing (if you use positional coding).


## Explicit difference between 'true division' and 'integer division'

While this change may be not a good fit for system programming in python, for data science this is definitely a positive change

```python
velocity = distance / time

data = pandas.read_csv('timing.csv')
velocity = data['distance'] / data['time']
```

Result in python2 depends on whether 'time' and 'distance' (e.g. measured in meters and seconds) are stored as integers.
In python3, result is correct in both cases, promotion. 

Another case is integer division, which is now an explicit operation:

```python
n_gifts = money // gift_price
```

## Constants in math module

```python
math.inf # 'largest' number
math.nan # not a number

max_quality = -math.inf

for model in trained_models:
    max_quality = max(max_quality, compute_quality(model, data))
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
  
if a: # very bad idea
  pass
```

## Iterable unpacking

```python
# handy when amount of additional stored info may vary between experiments, but the same code can be used in all cases
model_paramteres, optimizer_parameters, *other_params = load(checkpoint_name)

# picking two last values from a sequence
*prev, next_to_last, last = values_history

# This also works with any iterables, so if you have a function that yields e.g. qualities,
# below is a simple way to take only last two values from a list 
*prev, next_to_last, last = iter_train(args)
```

## Unicode 

```python
print(len('您好'))
```
Python2 outputs 6, python3 outputs 2. 

```
x = u'со'
x += 'со'
```
python2 fails, python3 works as expected (because I've used russian letters in this example).

In python3 `str`s are unicode strings, and it is more convenient for NLP processing of non-english texts.

There are less obvious things, for instance:
```python
print(sorted([u'a', 'a']))
print(sorted(['a', u'a']))
```

Python2 output:
```
[u'a', 'a']
['a', u'a']
```


## Default pickle engine provides better compression for arrays

```
import cPickle as pickle
import numpy
print len(pickle.dumps(numpy.random.normal(size=[1000, 1000])))
# prints 23691675

python 3
import pickle
import numpy
len(pickle.dumps(numpy.random.normal(size=[1000, 1000])))
# prints 8000162
```

You can actually achieve close compression with `protocol=2` parameter, but developers typically ignore this option (or simply not aware of it). 


## OrderedDict is faster now

OrderedDict is probably the most used structure after list. It a good old dictionary, which keeps the order in which keys were added. 


## Single integer type

Python2 provides two basic integer types, which are `int` (64-bit signed integer) and `long` (for long arithmetics).
Quite confusing after C++.

Python3 now has only `int`, which provides long arithmetics.

Checking for integer is easier in python 3:

```
isinstance(x, numbers.Integral) # python2, the canonical way
isinstance(x, [long, int]) # python2
isinstance(x, int) # python3, easiest to remember
```


## Other stuff

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
  Problem with iterators are:
  - no trivial slicing
  - no double usage
  
  Quite typically, you can resolve it by converting to list.

# Main problem for education

Data science courses will struggle with some of the changes.

Course authors should spend time in the beginning to explain what is an iterator, 
why is can't be sliced / concatenated like a string (and how to deal with it).

# Conclusion

Python3 is with us for almost 10 years, but right now it it time that you *should* move to python3.

There are issues with migration, but the advantages worth it.
Your research and production code should benefit significantly from moving to python3-only codebase.

And I can't wait the bright moment when libraries can drop support for python2 (which will happen quite soon) and completely enjoy new language features that were not backported.

Following migrations will be smoother: ["we will never do this kind of backwards-incompatible change again"](https://snarky.ca/why-python-3-exists/)

### Links

- http://sebastianraschka.com/Articles/2014_python_2_3_key_diff.html (и смотри внутри)

