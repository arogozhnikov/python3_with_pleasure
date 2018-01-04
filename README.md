# Migrating to Python 3 with pleasure
## A short guide on features of Python 3 for data scientists

Python became a mainstream language for machine learning and other scientific fields that heavily operate with data.
Python boasts various frameworks for deep learning and well-established set of tools for data processing and visualization.

However, Python ecosystem co-exists in Python 2 and Python 3, and Python 2 is still used among data scientists. 
By the end of 2019 scientific stack will [stop supporting Python2](http://www.python3statement.org).
As for numpy, after 2018 any new feature releases will support [only Python3](https://github.com/numpy/numpy/blob/master/doc/neps/dropping-python2.7-proposal.rst).

To make transition less frustrative, I've collected a bunch of Python 3 features that you may find useful
(and worth migrating faster). 


## Better paths handling with `pathlib`

`pathlib` is a default module in python3, that helps you to avoid tons of `os.path.join`s:

```python
from pathlib import Path

dataset = 'wiki_images'
datasets_root = Path('/path/to/datasets/') 

train_path = datasets_root / dataset / 'train'
test_path = datasets_root / dataset / 'test'

for image_path in train_path.iterdir():
    with image_path.open() as f: # note, open is a method of Path object
        # do something with an image
```

Previously it was always tempting to use string concatenation (which is concise, but obviously bad), 
with `pathlib` the code is safe, concise, and readable.

Also `pathlib.Path` has a bunch of methods, that every python novice had to google (and anyone who is not working with files all the time):

```python
p.exists()
p.is_dir()
p.parts()
p.with_name('sibling.png') # only change the name, but keep the folder
p.with_suffix('.jpg') # only change the extension, but keep the folder and the name
p.chmod(mode)
p.rmdir()
```

`pathlib` should save you lots of time, 
please see [docs](https://docs.python.org/3/library/pathlib.html) and [reference](https://pymotw.com/3/pathlib/) for more.


## Type hinting is now part of the language

Example of type hinting in pycharm: <br/>
<img src='pycharm-type-hinting.png' />

Python is not just a language for small scripts anymore, 
data pipelines these days include numerous steps each involving different frameworks (and sometimes very different logic).

Type hinting was introduced to help with growing complexity of programs, so machines could help with code verification.

For instance, the following code may work with dict, pandas.DataFrame, astropy.Frame, numpy.recarray and a dozen of other containers.

```python
def compute_time(data):
    data['time'] = data['distance'] / data['velocity'] 
```


Things are also quite complicated when operating with tensors, which may come from different frameworks.

```python
def convert_to_grayscale(images):
    return images.mean(axis=1)
```

Еще нужно return doctypes продемонстрировать, IDE контролирует, когда возвращается что-то не то.

```python
def train_on_batch(batch_data: tensor, batch_labels: tensor) -> Tuple[float, float]:
  ...
  loss = loss.mean()
  accuracy = (predicted == label).mean()
  return loss, accuracy
```

(In most cases) IDE will spot an error if you forgot to convert an accuracy to float.
If you're using dynamic graphs (with pytorch, chainer or somewhat alike), 
passing loss as tensor may also drive to memory overflow, because computation graph components would not be released.

If you have a significant codebase, hint tools like [MyPy](http://mypy.readthedocs.io) are likely to become part of your continuous integration pipeline. 

A webinar ["Putting Type Hints to Work"](https://www.youtube.com/watch?v=JqBCFfiE11g) by Daniel Pyrathon is good for a brief introduction.

Sidenote: unfortunately, right now hinting is not yet powerful enough to provide fine-grained typing for ndarrays/tensors, but [maybe we'll have it once](https://github.com/numpy/numpy/issues/7370), and this will be a great feature for DS.

## Type hinting → type checking in runtime

By default, type hinting does not influence how your code is working, but merely helps you to point code intentions.

However, you can enforce type checking in runtime with tools like ... [enforce](https://github.com/RussBaz/enforce), 
this can help you in debugging.

```python
@enforce.runtime_validation
def foo(text: str) -> None:
    print(text)

foo('Hi') # ok
foo(5)    # fails   

# enforce also supports callable arguments
@enforce.runtime_validation
def foo(a: typing.Callable[[int, int], str]) -> str:
    return a(5, 6)
```


## Matrix multiplication as @

Let's implement one of the simplest ML models &mdash; a linear regression with l2 regularization:

```python
# l2-regularized linear regression: || AX - b ||^2 + alpha * ||x||^2 -> min

# Python 2
X = np.linalg.inv(np.dot(A.T, A) + alpha * np.eye(A.shape[1])).dot(A.T.dot(b))
# Python 3
X = np.linalg.inv(A.T @ A + alpha * np.eye(A.shape[1])) @ (A.T @ b)
```

The code with `@` becomes more readable and more translatable between deep learning frameworks: same code `X @ W + b[None, :]` for a single layer of perceptron works in `numpy`, `cupy`, `pytorch`, `tensorflow` (and other frameworks that operate with tensors).


## Globbing with `**`

Recursive folder globbing is not easy in Python 2, even custom module [glob2](https://github.com/miracle2k/python-glob2) exists that overcomes this. Recursive flag is supported since Python 3.6:

```python
import glob
# Python 2

found_images = \
    glob.glob('/path/*.jpg') \
  + glob.glob('/path/*/*.jpg') \
  + glob.glob('/path/*/*/*.jpg') \
  + glob.glob('/path/*/*/*/*.jpg') \
  + glob.glob('/path/*/*/*/*/*.jpg') 

# Python 3

found_images = glob.glob('/path/**/*.jpg', recursive=True)
```

Better option is to use `pathlib` in python3 (minus one import!):
```python
# Python 3
found_images = pathlib.Path('/path/').glob('**/*.jpg')
```

## Print is a function now

Yes, code now has these annoying parentheses, but there are some advantages:

- simple syntax for using file descriptor:
    ```python
    # Python 3
    print >>sys.stderr, "critical error"      # Python 2
    print("critical error", file=sys.stderr)  # Python 3
    ```
- printing tab-aligned tables without `str.join`:
    ```python
    # Python 3
    print(*array, sep='\t')
    print(batch, epoch, loss, accuracy, time, sep='\t')
    ```
- hacky suppressing / redirection of printing output:
    ```python
    # Python 3
    _print = print # store the original print function
    def print(*args, **kargs):
        pass  # do something useful, e.g. store output to some file
    ```
    In jupyter it is desireable to log each output to a separate file (to track what's happening after you got disconnected), so you can override `print` now.
    
- `print` can participate in list comprehensions and other language constructs 
    ```python
    # Python 3
    result = process(x) if is_valid(x) else print('invalid item: ', x)
    ```

## f-strings for simple and reliable formatting

Default formatting system provides a flexibility that is not required in data experiments. 
Resulting code is either too verbose or too fragile towards any changes.

Quite typically data scientist outputs iteratively some logging information as in a fixed format. 
It is common to have a code like:

```python
# Python 2
print('{batch:3} {epoch:3} / {total_epochs:3}  accuracy: {acc_mean:0.4f}±{acc_std:0.4f} time: {avg_time:3.2f}'.format(
    batch=batch, epoch=epoch, total_epochs=total_epochs, 
    acc_mean=numpy.mean(accuracies), acc_std=numpy.std(accuracies),
    avg_time=time / len(data_batch)
))

# Python 2 (too error-prone during fast modifications, please avoid):
print('{:3} {:3} / {:3}  accuracy: {:0.4f}±{:0.4f} time: {:3.2f}'.format(
    batch, epoch, total_epochs, numpy.mean(accuracies), numpy.std(accuracies),
    time / len(data_batch)
))
```

Sample output:
```
120  12 / 300  accuracy: 0.8180±0.4649 time: 56.60
```

**f-strings** aka formatted string literals were introduced in Python 3.6:
```python
# Python 3.6+
print(f'{batch:3} {epoch:3} / {total_epochs:3}  accuracy: {numpy.mean(accuracies):0.4f}±{numpy.std(accuracies):0.4f} time: {time / len(data_batch):3.2f}')
```


## Explicit difference between 'true division' and 'integer division'

For data science this is definitely a handy change (but not for system programming, I believe)

```python
velocity = distance / time

data = pandas.read_csv('timing.csv')
velocity = data['distance'] / data['time']
```

Result in Python 2 depends on whether 'time' and 'distance' (e.g. measured in meters and seconds) are stored as integers.
In Python 3, result is correct in both cases, promotion to float happens automatically when needed. 

Another case is integer division, which is now an explicit operation:

```python
n_gifts = money // gift_price
```

Note, that this applies both to built-in types and to custom types provided by data packages (e.g. `numpy` or `pandas`).


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
# will throw an exception in Python 3
3 < '3'
2 < None
(3, 4) < (3, None)
(4, 5) < [4, 5]

# False in both Python 2 and Python 3
(4, 5) == [4, 5]
```

- prevents from occasional sorting of instances of different types
  ```python
  sorted([2, '1', 3])  # invalid for Python 3, in Python 2 returns [2, 3, '1']
  ```
- helps to spot some problems that arise when processing raw data

Sidenote: proper check for None is (in both Python versions)
```python
if a is not None:
  pass
  
if a: # WRONG check for None
  pass
```


## Unicode for NLP

```python
print(len('您好'))
```
Output:
- Python 2: `6`
- Python 3: `2`. 

```
x = u'со'
x += 'co' # ok
x += 'со' # fail
```
Python 2 fails, Python 3 works as expected (because I've used russian letters in strings).

In Python 3 `str`s are unicode strings, and it is more convenient for NLP processing of non-english texts.

There are less obvious funny things, for instance:

```python
print(sorted([u'a', 'a']))
print(sorted(['a', u'a']))
```

Python 2 outputs:
```
[u'a', 'a']
['a', u'a']
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

## Default pickle engine provides better compression for arrays

```python
# Python 2
import cPickle as pickle
import numpy
print len(pickle.dumps(numpy.random.normal(size=[1000, 1000])))
# result: 23691675

# Python 3
import pickle
import numpy
len(pickle.dumps(numpy.random.normal(size=[1000, 1000])))
# result: 8000162
```

Three times less space. And it is *much* faster.
Actually similar compression (but not speed) is achievable with `protocol=2` parameter, but users typically ignore this option (or simply not aware of it). 


## Safer comprehensions

```python
labels = <initial_value>
predictions = [model.predict(data) for data, labels in dataset]

# labels are overwritten in Python 2
# labels are not affected by comprehension in Python 3
```

## Single integer type

Python 2 provides two basic integer types, which are `int` (64-bit signed integer) and `long` for long arithmetics (quite confusing after C++).

Python 3 has a single type `int`, which provides long arithmetics as well.

Checking for integer:

```
isinstance(x, numbers.Integral) # Python 2, the canonical way
isinstance(x, [long, int])      # Python 2
isinstance(x, int)              # Python 3, easier to remember
```

## Other stuff

- `Enum`s
- yield from 
- async / await
- keyword-only arguments  `def f(a, b, *, option=True):` allows much [simpler creation of 'future-proof APIs'](http://www.asmeurer.com/python3-presentation/slides.html#12)
- some libraries e.g. [jupyterhub](https://github.com/jupyterhub/jupyterhub) (jupyter in cloud) only support Python 3.4


## Main problems for code in data science and how to resolve those

- relative imports from subdirectories
  - packaging 
  - sys.path.insert
  - softlinks
  
- support for nested arguments [was dropped](https://www.python.org/dev/peps/pep-3113/)
  ```
  map(lambda x, (y, z): x, z, dict.items())
  ```
  
  However, it is still perfectly working with different comprehensions:
  ```python
  {x:z for x, (y, z) in d.items()}
  ```
  In general, comprehensions are also better 'translatable' between Python 2 and 3.

- `map`, `.values()`, `.items()` do not return lists.
  Problem with iterators are:
  - no trivial slicing
  - can't be used twice
  
  Almost all of the problems are resolved by converting result to list.


## Main problems for teaching machine learning and data science with python 

Data science courses struggle with some of the changes (but python is still the most reasonable option).

Course authors should spend time in the beginning to explain what is an iterator, 
why is can't be sliced / concatenated like a string (and how to deal with it).


# Conclusion

Python 2 and Python 3 co-exist for almost 10 years, but right now it is time that we *should* move to Python 3.

There are issues with migration, but the advantages worth it.
Your research and production code should become safer, shorter, and more readable after moving to Python 3-only codebase.

And I can't wait for the bright moment when libraries drop support for Python 2 and completely enjoy new language features.

Following migrations are promised to be smoother: ["we will never do this kind of backwards-incompatible change again"](https://snarky.ca/why-python-3-exists/)

### Links

- [Key differences between Python 2.7 and Python 3.x](http://sebastianraschka.com/Articles/2014_python_2_3_key_diff.html) (и смотри внутри)
- [Python FAQ: How do I port to Python 3?](https://eev.ee/blog/2016/07/31/python-faq-how-do-i-port-to-python-3/)
- [10 awesome features of Python that you can't use because you refuse to upgrade to Python 3](http://www.asmeurer.com/python3-presentation/slides.html)
