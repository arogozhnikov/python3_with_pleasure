# 快乐迁移Python 3
## 为数据科学家提供的关于Python 3特性的简介

> Python became a mainstream language for machine learning and other scientific fields that heavily operate with data;
it boasts various deep learning frameworks and well-established set of tools for data processing and visualization.

Python 已成为机器学习以及其他紧密结合数据的科学领域的主流语言；它提供了各种深度学习的框架以及一系列完善的数据处理和可视化工具。

> However, Python ecosystem co-exists in Python 2 and Python 3, and Python 2 is still used among data scientists.
By the end of 2019 the scientific stack will [stop supporting Python2](http://www.python3statement.org).
As for numpy, after 2018 any new feature releases will only support [Python3](https://github.com/numpy/numpy/blob/master/doc/neps/dropping-python2.7-proposal.rst).

然而，Python 的生态圈中 Python 2 和 Python 3 是共存状态，并且数据科学家之中是依然有使用 Python 2 的。2019年年底（Python的）科学组件将会[停止支持 Python 2 ](http://www.python3statement.org)。 至于numpy，2018年之后任何推出的新特性将会只支持[Python 3](https://github.com/numpy/numpy/blob/master/doc/neps/dropping-python2.7-proposal.rst) 。

>To make the transition less frustrating, I've collected a bunch of Python 3 features that you may find useful.

为了让这一过渡更轻松一点，我整理了一些 Python 3 你可能觉得有用的特性。

<img src='https://uploads.toptal.io/blog/image/92216/toptal-blog-image-1457618659472-be2f380fe3aad41333427ecd5a1ec5c5.jpg' width=400 />

图片来源 [Dario Bertini post (toptal)](https://www.toptal.com/python/python-3-is-it-worth-the-switch)

## `pathlib`提供了更好的路径处理

> `pathlib` is a default module in python3, that helps you to avoid tons of `os.path.join`s:

`pathlib` 是Python 3 一个默认的组件，有助于避免大量使用`os.path.join`：

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

> Previously it was always tempting to use string concatenation (concise, but obviously bad),
now with `pathlib` the code is safe, concise, and readable.

以前，人们倾向于使用字符串连接（虽然简洁，但明显不好）；现在，代码中用`pathlib`是安全的，简洁的，并且更有可读性。

> Also `pathlib.Path` has a bunch of methods and properties, that every python novice previously had to google:

此外，`pathlib.Path`有大量的方法和属性，每一位 Python 早期的初学者不得不谷歌了解：

```python
p.exists()
p.is_dir()
p.parts
p.with_name('sibling.png') # only change the name, but keep the folder
p.with_suffix('.jpg') # only change the extension, but keep the folder and the name
p.chmod(mode)
p.rmdir()
```

> `pathlib` should save you lots of time,
please see [docs](https://docs.python.org/3/library/pathlib.html) and [reference](https://pymotw.com/3/pathlib/) for more.

`pathlib` 应当会节省大量时间，请参看[文档](https://docs.python.org/3/library/pathlib.html)以及[指南](https://pymotw.com/3/pathlib/)了解更多。


## 类型提示现在已是这语言的一部分

> Example of type hinting in pycharm: <br/>

pycharm环境类型提示的例子：

<img src='images/pycharm-type-hinting.png' />

> Python is not just a language for small scripts anymore,
data pipelines these days include numerous steps each involving different frameworks (and sometimes very different logic).

Python 不再是一种小型的脚本语言，数据管道现如今包含数个级别，而每一级又涉及到不同的框架（甚至有时是千差万别的逻辑）。

> Type hinting was introduced to help with growing complexity of programs, so machines could help with code verification.
Previously different modules used custom ways to point [types in docstrings](https://www.jetbrains.com/help/pycharm/type-hinting-in-pycharm.html#legacy)
(Hint: pycharm can convert old docstrings to fresh type hinting).

类型提示的引入是为了在程序的持续增加的复杂性方面提供帮助，这样机器可以辅助代码验证。以前不同的模块使用自定义的方式指定[文档字符中的类型](https://www.jetbrains.com/help/pycharm/type-hinting-in-pycharm.html#legacy)(提示：pycharm可以将旧的字符串转换成新的类型提示)。

> As a simple example, the following code may work with different types of data (that's what we like about python data stack).

作为一个简单的例子，下面的代码可以适用于数据的不同类型（这也是关于数据栈我们喜欢的一点）。

```python
def repeat_each_entry(data):
    """ Each entry in the data is doubled
    <blah blah nobody reads the documentation till the end>
    """
    index = numpy.repeat(numpy.arange(len(data)), 2)
    return data[index]
```

> This code e.g. works for `numpy.array` (incl. multidimensional ones), `astropy.Table` and `astropy.Column`, `bcolz`, `cupy`, `mxnet.ndarray` and others.

这段代码可适用于例如 `numpy.array` (包括多维数组)， `astropy.Table` 以及 `astropy.Column`， `bcolz`， `cupy`, `mxnet.ndarray` 和其他的组件。

> This code will work for `pandas.Series`, but in the wrong way:

这段代码虽然也适用于`pandas.Series`，但是是错误的使用方式：

```python
repeat_each_entry(pandas.Series(data=[0, 1, 2], index=[3, 4, 5])) # returns Series with Nones inside
```

> This was two lines of code. Imagine how unpredictable behavior of a complex system, because just one function may misbehave.
Stating explicitly which types a method expects is very helpful in large systems, this will warn you if a function was passed unexpected arguments.

这曾经是两行代码。想象一下一个复杂系统不可预知的行为，仅仅是因为一个功能可能会失败。在大型的系统中，明确地指出方法期望的类型是非常有帮助的。如果一个方法通过了意外参数，则会给出警告。

```python
def repeat_each_entry(data: Union[numpy.ndarray, bcolz.carray]):
```
> If you have a significant codebase, hinting tools like [MyPy](http://mypy.readthedocs.io) are likely to become part of your continuous integration pipeline.A webinar ["Putting Type Hints to Work"](https://www.youtube.com/watch?v=JqBCFfiE11g) by Daniel Pyrathon is good for a brief introduction.

如果你有一个重要的代码仓库，比如[MyPy](http://mypy.readthedocs.io)的提示工具有可能成为你持续集成管道的一部分。Daniel Pyrathon主持的["Putting Type Hints to Work"](https://www.youtube.com/watch?v=JqBCFfiE11g)研讨会，给出了一个很好的简介。

> Sidenote: unfortunately, hinting is not yet powerful enough to provide fine-grained typing for ndarrays/tensors, but [maybe we'll have it once](https://github.com/numpy/numpy/issues/7370), and this will be a great feature for DS.

边注：不幸的是，提示信息还不够强大为多维数组/张量提供精细的提示。但是[也许我们会有](https://github.com/numpy/numpy/issues/7370)，并且这将是DS的一个强大功能。

## 类型提示 → 在运行时检查类型

> By default, function annotations do not influence how your code is working, but merely help you to point code intentions.

默认情况下，方法声明不会影响你运行中的代码，而只是帮助你指出代码的意图。

> However, you can enforce type checking in runtime with tools like ... [enforce](https://github.com/RussBaz/enforce),
this can help you in debugging (there are many cases when type hinting is not working).

然而，你可以利用工具，比如[enforce](https://github.com/RussBaz/enforce)，在代码运行时执行类型检查，这对你在debug代码时是很有帮助的（类型提示不起作用的情况也很多）。

```python
@enforce.runtime_validation
def foo(text: str) -> None:
    print(text)

foo('Hi') # ok
foo(5)    # fails


@enforce.runtime_validation
def any2(x: List[bool]) -> bool:
    return any(x)

any ([False, False, True, False]) # True
any2([False, False, True, False]) # True

any (['False']) # True
any2(['False']) # fails

any ([False, None, "", 0]) # False
any2([False, None, "", 0]) # fails

```

## 方法声明的其他用途

> As mentioned before, annotations do not influence code execution, but rather provide some meta-information,
and you can use it as you wish.

正如之前提到的，声明不会影响代码执行，而只是提供一些元信息，此外你也可以随意使用。

> For instance, measurement units are a common pain in scientific areas, `astropy` package [provides a simple decorator](http://docs.astropy.org/en/stable/units/quantity.html#functions-that-accept-quantities) to control units of input quantities and convert output to required units.

比如，测量单位是科学领域常见的痛点，`astropy`包[提供了一个简单的装饰器](http://docs.astropy.org/en/stable/units/quantity.html#functions-that-accept-quantities)用来控制输入数量的单位及转换输出部分所需的单位。
```python
# Python 3
from astropy import units as u
@u.quantity_input()
def frequency(speed: u.meter / u.s, wavelength: u.m) -> u.terahertz:
    return speed / wavelength

frequency(speed=300_000 * u.km / u.s, wavelength=555 * u.nm)
# output: 540.5405405405404 THz, frequency of green visible light
```

> If you're processing tabular scientific data in python (not necessarily astronomical), you should give `astropy` a shot.

如果你正在用Python处理表格式的科学数据（没必要是天文数字），那么你应该试试`astropy`。

> You can also define your application-specific decorators to perform control / conversion of inputs and output in the same manner.

你也可以自定义专用的装饰器，以相同的方式执行输入和输出的控制/转换。

## 矩阵乘号 @ 

> Let's implement one of the simplest ML models &mdash; a linear regression with l2 regularization (a.k.a. ridge regression):

让我们来实现一个最简单的 ML(机器学习) 模型 &mdash; 具有 l2 正则化的线性回归（又名岭回归）：

```python
# l2-regularized linear regression: || AX - b ||^2 + alpha * ||x||^2 -> min

# Python 2
X = np.linalg.inv(np.dot(A.T, A) + alpha * np.eye(A.shape[1])).dot(A.T.dot(b))
# Python 3
X = np.linalg.inv(A.T @ A + alpha * np.eye(A.shape[1])) @ (A.T @ b)
```

> The code with `@` becomes more readable and more translatable between deep learning frameworks: same code `X @ W + b[None, :]` for a single layer of perceptron works in `numpy`, `cupy`, `pytorch`, `tensorflow` (and other frameworks that operate with tensors).

使用`@`的代码在深度学习框架之间变得更有可读性和可转换性：对于单层感知器，相同的代码`X @ W + b[None, :]` 可运行与`numpy`、 `cupy`、 `pytorch`、 `tensorflow`（以及其他基于张量运行的框架）。

## 通配符 `**`

> Recursive folder globbing is not easy in Python 2, even though the [glob2](https://github.com/miracle2k/python-glob2) custom module exists that overcomes this. A recursive flag is supported since Python 3.5:

即使[glob2](https://github.com/miracle2k/python-glob2)的自定义模块克服了这一点，但是在Python 2中递归的文件夹通配依旧不容易。自Python3.5以来便支持了递归标志：

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

> A better option is to use `pathlib` in python3 (minus one import!):

一个更好的选项就是在Python 3中使用`pathlib`(减少了一个导入！)：
```python
# Python 3
found_images = pathlib.Path('/path/').glob('**/*.jpg')
```

## Print 现在成了一个方法

> Yes, code now has these annoying parentheses, but there are some advantages:

是的，代码现在有了这些烦人的括号，但也是有一些好处的：

> - simple syntax for using file descriptor:
- 使用文件描述符的简单语法:

    ```python
    print >>sys.stderr, "critical error"      # Python 2
    print("critical error", file=sys.stderr)  # Python 3
    ```
> - printing tab-aligned tables without `str.join`:
- 不使用`str.join`打印制表符对齐表：

    ```python
    # Python 3
    print(*array, sep='\t')
    print(batch, epoch, loss, accuracy, time, sep='\t')
    ```
> - hacky suppressing / redirection of printing output:
- 结束/重定向打印输出

    ```python
    # Python 3
    _print = print # store the original print function
    def print(*args, **kargs):
        pass  # do something useful, e.g. store output to some file
    ```
    > In jupyter it is desirable to log each output to a separate file (to track what's happening after you got disconnected), so you can override `print` now.
    
    在jupyter中，最好将每个输出记录到一个单独的文件中（以便跟踪断开连接后发生的情况），以便你现在可以重写`print`。

    > Below you can see a context manager that temporarily overrides behavior of print:
    
    下面你可以看到暂时覆盖打印行为的上下文管理器：

    ```python
    @contextlib.contextmanager
    def replace_print():
        import builtins
        _print = print # saving old print function
        # or use some other function here
        builtins.print = lambda *args, **kwargs: _print('new printing', *args, **kwargs)
        yield
        builtins.print = _print

    with replace_print():
        <code here will invoke other print function>
    ```
    > It is *not* a recommended approach, but a small dirty hack that is now possible.
    
    这*并不是*推荐的方法，现在却可能是一次小小的黑客攻击。
> - `print` can participate in list comprehensions and other language constructs
- `print`可以参与列表理解和其他语言结构:

    ```python
    # Python 3
    result = process(x) if is_valid(x) else print('invalid item: ', x)
    ```


## 数字中的下划线 （千位分隔符）

[PEP-515](https://www.python.org/dev/peps/pep-0515/ "PEP-515") introduced underscores in Numeric Literals.
In Python3, underscores can be used to group digits visually in integral, floating-point, and complex number literals.

```python
# grouping decimal numbers by thousands
one_million = 1_000_000

# grouping hexadecimal addresses by words
addr = 0xCAFE_F00D

# grouping bits into nibbles in a binary literal
flags = 0b_0011_1111_0100_1110

# same, for string conversions
flags = int('0b_1111_0000', 2)
```

## f-strings for simple and reliable formatting

The default formatting system provides a flexibility that is not required in data experiments.
The resulting code is either too verbose or too fragile towards any changes.

Quite typically data scientists outputs some logging information iteratively in a fixed format.
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
data = pandas.read_csv('timing.csv')
velocity = data['distance'] / data['time']
```

Results in Python 2 depend on whether 'time' and 'distance' (e.g. measured in meters and seconds) are stored as integers.
In Python 3, the result is correct in both cases, because the result of division is float.

Another case is integer division, which is now an explicit operation:

```python
n_gifts = money // gift_price  # correct for int and float arguments
```

Note, that this applies both to built-in types and to custom types provided by data packages (e.g. `numpy` or `pandas`).


## Strict ordering

```python
# All these comparisons are illegal in Python 3
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
s = '您好'
print(len(s))
print(s[:2])
```
Output:
- Python 2: `6\n��`
- Python 3: `2\n您好`.

```
x = u'со'
x += 'co' # ok
x += 'со' # fail
```
Python 2 fails, Python 3 works as expected (because I've used russian letters in strings).

In Python 3 `str`s are unicode strings, and it is more convenient for NLP processing of non-english texts.

There are other funny things, for instance:
```python
'a' < type < u'a'  # Python 2: True
'a' < u'a'         # Python 2: False
```

```python
from collections import Counter
Counter('Möbelstück')
```

- Python 2: `Counter({'\xc3': 2, 'b': 1, 'e': 1, 'c': 1, 'k': 1, 'M': 1, 'l': 1, 's': 1, 't': 1, '\xb6': 1, '\xbc': 1})`
- Python 3: `Counter({'M': 1, 'ö': 1, 'b': 1, 'e': 1, 'l': 1, 's': 1, 't': 1, 'ü': 1, 'c': 1, 'k': 1})`

You can handle all of this in Python 2 properly, but Python 3 is more friendly.

## Preserving order of dictionaries and **kwargs

In CPython 3.6+ dicts behave like `OrderedDict` by default (and [this is guaranteed in Python 3.7+](https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6)).
This preserves order during dict comprehensions (and other operations, e.g. during json serialization/deserialization)

```python
import json
x = {str(i):i for i in range(5)}
json.loads(json.dumps(x))
# Python 2
{u'1': 1, u'0': 0, u'3': 3, u'2': 2, u'4': 4}
# Python 3
{'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
```

Same applies to `**kwargs` (in Python 3.6+), they're kept in the same order as they appear in parameters.
Order is crucial when it comes to data pipelines, previously we had to write it in a cumbersome manner:
```
from torch import nn

# Python 2
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))

# Python 3.6+, how it *can* be done, not supported right now in pytorch
model = nn.Sequential(
    conv1=nn.Conv2d(1,20,5),
    relu1=nn.ReLU(),
    conv2=nn.Conv2d(20,64,5),
    relu2=nn.ReLU())
)
```

Did you notice? Uniqueness of names is also checked automatically.


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
Actually similar compression (but not speed) is achievable with `protocol=2` parameter, but users typically ignore this option (or simply are not aware of it).


## Safer comprehensions

```python
labels = <initial_value>
predictions = [model.predict(data) for data, labels in dataset]

# labels are overwritten in Python 2
# labels are not affected by comprehension in Python 3
```

## Super, simply super()

Python 2 `super(...)` was a frequent source of mistakes in code.

```python
# Python 2
class MySubClass(MySuperClass):
    def __init__(self, name, **options):
        super(MySubClass, self).__init__(name='subclass', **options)

# Python 3
class MySubClass(MySuperClass):
    def __init__(self, name, **options):
        super().__init__(name='subclass', **options)
```

More on `super` and method resolution order on [stackoverflow](https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods).

## Better IDE suggestions with variable annotations

The most enjoyable thing about programming in languages like Java, C# and alike is that IDE can make very good suggestions,
because type of each identifier is known before executing a program.

In python this is hard to achieve, but annotations will help you
- write your expectations in a clear form
- and get good suggestions from IDE

<img src='images/variable_annotations.png' /><br />
This is an example of PyCharm suggestions with variable annotations.
This works even in situations when functions you use are not annotated (e.g. due to backward compatibility).

## Multiple unpacking

Here is how you merge two dicts now:
```python
x = dict(a=1, b=2)
y = dict(b=3, d=4)
# Python 3.5+
z = {**x, **y}
# z = {'a': 1, 'b': 3, 'd': 4}, note that value for `b` is taken from the latter dict.
```

See [this thread at StackOverflow](https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression) for a comparison with Python 2.

The same approach also works for lists, tuples, and sets (`a`, `b`, `c` are any iterables):
```python
[*a, *b, *c] # list, concatenating
(*a, *b, *c) # tuple, concatenating
{*a, *b, *c} # set, union
```

Functions also [support this](https://docs.python.org/3/whatsnew/3.5.html#whatsnew-pep-448) for `*args` and `**kwargs`:
```
Python 3.5+
do_something(**{**default_settings, **custom_settings})

# Also possible, this code also checks there is no intersection between keys of dictionaries
do_something(**first_args, **second_args)
```

## Future-proof APIs with keyword-only arguments

Let's consider this snippet
```python
model = sklearn.svm.SVC(2, 'poly', 2, 4, 0.5)
```
Obviously, an author of this code didn't get the Python style of coding yet (most probably, just jumped from cpp or rust).
Unfortunately, this is not just question of taste, because changing the order of arguments (adding/deleting) in `SVC` will break this code. In particular, `sklearn` does some reordering/renaming from time to time of numerous algorithm parameters to provide consistent API. Each such refactoring may drive to broken code.

In Python 3, library authors may demand explicitly named parameters by using `*`:
```
class SVC(BaseSVC):
    def __init__(self, *, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, ... )
```
- users have to specify names of parameters `sklearn.svm.SVC(C=2, kernel='poly', degree=2, gamma=4, coef0=0.5)` now
- this mechanism provides a great combination of reliability and flexibility of APIs


## Minor: constants in `math` module

```python
# Python 3
math.inf # 'largest' number
math.nan # not a number

max_quality = -math.inf  # no more magic initial values!

for model in trained_models:
    max_quality = max(max_quality, compute_quality(model, data))
```

## Minor: single integer type

Python 2 provides two basic integer types, which are `int` (64-bit signed integer) and `long` for long arithmetics (quite confusing after C++).

Python 3 has a single type `int`, which incorporates long arithmetics.

Here is how you check that value is integer:

```
isinstance(x, numbers.Integral) # Python 2, the canonical way
isinstance(x, (long, int))      # Python 2
isinstance(x, int)              # Python 3, easier to remember
```

## Other stuff

- `Enum`s are theoretically useful, but
    - string-typing is already widely adopted in the python data stack
    - `Enum`s don't seem to interplay with numpy and categorical from pandas
- coroutines also *sound* very promising for data pipelining (see [slides](http://www.dabeaz.com/coroutines/Coroutines.pdf) by David Beazley), but I don't see their adoption in the wild.
- Python 3 has [stable ABI](https://www.python.org/dev/peps/pep-0384/)
- Python 3 supports unicode identifies (so `ω = Δφ / Δt` is ok), but you'd [better use good old ASCII names](https://stackoverflow.com/a/29855176/498892)
- some libraries e.g. [jupyterhub](https://github.com/jupyterhub/jupyterhub) (jupyter in cloud), django and fresh ipython only support Python 3, so features that sound useless for you are useful for libraries you'll probably want to use once.


### Problems for code migration specific for data science (and how to resolve those)

> - support for nested arguments [was dropped](https://www.python.org/dev/peps/pep-3113/)
  ```
  map(lambda x, (y, z): x, z, dict.items())
  ```

>  However, it is still perfectly working with different comprehensions:
  ```python
  {x:z for x, (y, z) in d.items()}
  ```
>  In general, comprehensions are also better 'translatable' between Python 2 and 3.

> - `map()`, `.keys()`, `.values()`, `.items()`, etc. return iterators, not lists. Main problems with iterators are:
   - no trivial slicing
   - can't be iterated twice

>  Almost all of the problems are resolved by converting result to list.

> - see [Python FAQ: How do I port to Python 3?](https://eev.ee/blog/2016/07/31/python-faq-how-do-i-port-to-python-3/) when in trouble

### Main problems for teaching machine learning and data science with python

> Course authors should spend time in the first lectures to explain what is an iterator,
why it can't be sliced / concatenated / multiplied / iterated twice like a string (and how to deal with it).

> I think most course authors would be happy to avoid these details, but now it is hardly possible.

# 结论

> Python 2 and Python 3 have co-existed for almost 10 years, but we *should* move to Python 3.

虽然Python 2 和 Python 3 已经共存了十年有余，但是我们*应该*要过渡到Python 3 了。

> Research and production code should become a bit shorter, more readable, and significantly safer after moving to Python 3-only codebase.

在转向使用唯一的 Python 3 代码库之后，研究和生产的代码将会变得更剪短，更有可读性，以及明显是更加安全的。

> Right now most libraries support both Python versions.
And I can't wait for the bright moment when packages drop support for Python 2 and enjoy new language features.

目前大部分类库都会支持两个Python版本，我已等不及要使用新的语言特性了，也同样期待依赖包舍弃对 Python 2 支持这一光明时刻的到来。

> Following migrations are promised to be smoother: ["we will never do this kind of backwards-incompatible change again"](https://snarky.ca/why-python-3-exists/)

以后的(版本)迁移会更加顺利：[我们再也不会做这种不向后兼容的变化了](https://snarky.ca/why-python-3-exists/)。

### 相关链接

- [Key differences between Python 2.7 and Python 3.x](http://sebastianraschka.com/Articles/2014_python_2_3_key_diff.html)
- [Python FAQ: How do I port to Python 3?](https://eev.ee/blog/2016/07/31/python-faq-how-do-i-port-to-python-3/)
- [10 awesome features of Python that you can't use because you refuse to upgrade to Python 3](http://www.asmeurer.com/python3-presentation/slides.html)
- [Trust me, python 3.3 is better than 2.7 (video)](http://pyvideo.org/pycon-us-2013/python-33-trust-me-its-better-than-27.html)
- [Python 3 for scientists](http://python-3-for-scientists.readthedocs.io/en/latest/)


### 版权声明

This text was published by [Alex Rogozhnikov](https://arogozhnikov.github.io/about/) under [CC BY-SA 3.0 License](https://creativecommons.org/licenses/by-sa/3.0/) (excluding images).

Translated to Chinese by Hunter-liu (@lq920320).


