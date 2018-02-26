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
- 结束/重定向打印输出：
    ```python
    # Python 3
    _print = print # store the original print function
    def print(*args, **kargs):
        pass  # do something useful, e.g. store output to some file
    ```
    In jupyter it is desirable to log each output to a separate file (to track what's happening after you got disconnected), so you can override `print` now.

    在jupyter中，最好将每个输出记录到一个单独的文件中（以便跟踪断开连接后发生的情况），以便你现在可以重写 `print` 。

    Below you can see a context manager that temporarily overrides behavior of print:

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
    It is *not* a recommended approach, but a small dirty hack that is now possible.

    这*并不是*推荐的方法，现在却可能是一次小小的黑客攻击。
> - `print` can participate in list comprehensions and other language constructs
- `print` 可以参与理解列表和其他语言结构:

    ```python
    # Python 3
    result = process(x) if is_valid(x) else print('invalid item: ', x)
    ```


## 数字中的下划线 （千位分隔符）

> [PEP-515](https://www.python.org/dev/peps/pep-0515/ "PEP-515") introduced underscores in Numeric Literals.
In Python3, underscores can be used to group digits visually in integral, floating-point, and complex number literals.

[PEP-515](https://www.python.org/dev/peps/pep-0515/ "PEP-515")在数字中引入了下划线。在Python 3 中，下划线可以用于在整数，浮点数，以及一些复杂的数字中以可视的方式对数字分组。

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

## 用于简单可靠格式化的 f-strings

> The default formatting system provides a flexibility that is not required in data experiments.
The resulting code is either too verbose or too fragile towards any changes.

默认的格式化系统提供了数据实验中不必要的灵活性。由此产生的代码对于任何更改都显得过于冗长或者脆弱。

> Quite typically data scientists outputs some logging information iteratively in a fixed format.
It is common to have a code like:

通常数据科学家会以固定的格式反复输出一些记录信息。如下代码就是常见的一段：

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

简单输出:
```
120  12 / 300  accuracy: 0.8180±0.4649 time: 56.60
```

> **f-strings** aka formatted string literals were introduced in Python 3.6:

**f-string** 又名格式化的字符串，在Python 3.6 中引入：
```python
# Python 3.6+
print(f'{batch:3} {epoch:3} / {total_epochs:3}  accuracy: {numpy.mean(accuracies):0.4f}±{numpy.std(accuracies):0.4f} time: {time / len(data_batch):3.2f}')
```


## “真正的除法”与“整数除法”之间的明显区别

> For data science this is definitely a handy change.

对于数据科学来说，这绝对是一个便利的改变。

```python
data = pandas.read_csv('timing.csv')
velocity = data['distance'] / data['time']
```

> Results in Python 2 depend on whether 'time' and 'distance' (e.g. measured in meters and seconds) are stored as integers.
In Python 3, the result is correct in both cases, because the result of division is float.

Python 2 中的计算结果取决于“时间”和“距离”（例如，分别以米和秒计量）是否存储为整数，而在Python 3 中，结果在两种情况下都是正确的，因为除法的计算结果是浮点型了。

> Another case is integer division, which is now an explicit operation:

另一种情况是整数除法，它现在是一种精确的运算了：

```python
n_gifts = money // gift_price  # correct for int and float arguments
```

> Note, that this applies both to built-in types and to custom types provided by data packages (e.g. `numpy` or `pandas`).

注意，这都适用于内置类型及数据包提供的自定义类型（如`numpy` 或者 `pandas`）。

## 严谨的排序

```python
# All these comparisons are illegal in Python 3
3 < '3'
2 < None
(3, 4) < (3, None)
(4, 5) < [4, 5]

# False in both Python 2 and Python 3
(4, 5) == [4, 5]
```

> - prevents from occasional sorting of instances of different types
- 防止偶尔对不同类型的实例进行排序
  ```python
  sorted([2, '1', 3])  # invalid for Python 3, in Python 2 returns [2, 3, '1']
  ```
> - helps to spot some problems that arise when processing raw data
- 有助于发现在处理原始数据时的一些问题

> Sidenote: proper check for None is (in both Python versions)

边注：合理检查None的情况（Python两个版本中都有）
```python
if a is not None:
  pass

if a: # WRONG check for None
  pass
```


## 用于NLP的Unicode 

*译者注：NLP，神经语言程序学 (Neuro-Linguistic Programming) *

```python
s = '您好'
print(len(s))
print(s[:2])
```
输出:
- Python 2: `6\n��`
- Python 3: `2\n您好`.

```
x = u'со'
x += 'co' # ok
x += 'со' # fail
```
> Python 2 fails, Python 3 works as expected (because I've used russian letters in strings).

Python 2 失败了，Python 3 如预期运行（因为我在字符串中使用了俄语的文字）。

> In Python 3 `str`s are unicode strings, and it is more convenient for NLP processing of non-english texts.

在Python 3 中，`str`是unicode字符串，对于非英文文本的NLP处理更为方便。

> There are other funny things, for instance:

这还有一些其他有趣的事情，比如：
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

> You can handle all of this in Python 2 properly, but Python 3 is more friendly.

虽然你可以用Python 2正确地处理所有这些情况，但Python 3显得更加友好。

## 保留字典和** kwargs的顺序

> In CPython 3.6+ dicts behave like `OrderedDict` by default (and [this is guaranteed in Python 3.7+](https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6)).
This preserves order during dict comprehensions (and other operations, e.g. during json serialization/deserialization)

在CPython 3.6+中，字典的默认行为与`OrderedDict`类似（并且[这在Python 3.7+ 中也得到了保证]((https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6))）。这在字典释义时提供了顺序（以及其他操作执行时，比如json序列化/反序列化）。
```python
import json
x = {str(i):i for i in range(5)}
json.loads(json.dumps(x))
# Python 2
{u'1': 1, u'0': 0, u'3': 3, u'2': 2, u'4': 4}
# Python 3
{'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
```

> Same applies to `**kwargs` (in Python 3.6+), they're kept in the same order as they appear in parameters.
Order is crucial when it comes to data pipelines, previously we had to write it in a cumbersome manner:

同样适用于`** kwargs`（Python 3.6+），它们保持与它们在参数中出现的顺序相同。在数据管道方面，顺序至关重要，以前我们必须以繁琐的方式来编写：
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

> Did you notice? Uniqueness of names is also checked automatically.

你注意到了吗？命名的唯一性也会自动检查。


## 可迭代对象的（Iterable）解压

```python
# handy when amount of additional stored info may vary between experiments, but the same code can be used in all cases
model_paramteres, optimizer_parameters, *other_params = load(checkpoint_name)

# picking two last values from a sequence
*prev, next_to_last, last = values_history

# This also works with any iterables, so if you have a function that yields e.g. qualities,
# below is a simple way to take only last two values from a list
*prev, next_to_last, last = iter_train(args)
```

## 默认的pickle引擎为数组提供更好的压缩

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

> Three times less space. And it is *much* faster.
Actually similar compression (but not speed) is achievable with `protocol=2` parameter, but users typically ignore this option (or simply are not aware of it).

1/3的空间，以及*更加*快的速度。事实上，使用`protocol = 2`参数可以实现类似的压缩（速度则大相径庭），但用户通常会忽略此选项（或者根本不知道它）。


## 更安全的压缩

```python
labels = <initial_value>
predictions = [model.predict(data) for data, labels in dataset]

# labels are overwritten in Python 2
# labels are not affected by comprehension in Python 3
```

## 超简单的super()函数

> Python 2 `super(...)` was a frequent source of mistakes in code.

Python 2 中的`super(...)`曾是代码中最常见的错误源头。

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

> More on `super` and method resolution order on [stackoverflow](https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods).

[stackoverflow](https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods)上有更多关于`super`和方法解决的信息。

## 有着变量注释的更好的IDE建议

> The most enjoyable thing about programming in languages like Java, C# and alike is that IDE can make very good suggestions,
because type of each identifier is known before executing a program.

关于Java，C＃等语言编程最令人享受的事情是IDE可以提出非常好的建议，因为每个标识符的类型在执行程序之前是已知的。

> In python this is hard to achieve, but annotations will help you
> - write your expectations in a clear form
> - and get good suggestions from IDE

Python中这很难实现，但注释是会帮助你的
- 以清晰的形式写下你的期望
- 并从IDE获得很好的建议

<img src='images/variable_annotations.png' /><br />
> This is an example of PyCharm suggestions with variable annotations.
This works even in situations when functions you use are not annotated (e.g. due to backward compatibility).

这是PyCharm带有变量声明建议的一个例子。即使在你使用的功能未被注释过的情况依旧是有效的（例如，向后的兼容性）。

## 更多的解包（unpacking）

> Here is how you merge two dicts now:

现在展示如何合并两个字典：
```python
x = dict(a=1, b=2)
y = dict(b=3, d=4)
# Python 3.5+
z = {**x, **y}
# z = {'a': 1, 'b': 3, 'd': 4}, note that value for `b` is taken from the latter dict.
```

> See [this thread at StackOverflow](https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression) for a comparison with Python 2.

请参照[在StackOverflow上的这一过程](https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression)，与Python 2进行比较。

> The same approach also works for lists, tuples, and sets (`a`, `b`, `c` are any iterables):

同样的方法对于列表，元组，以及集合（`a`, `b`, `c` 是可任意迭代的）：
```python
[*a, *b, *c] # list, concatenating
(*a, *b, *c) # tuple, concatenating
{*a, *b, *c} # set, union
```

> Functions also [support this](https://docs.python.org/3/whatsnew/3.5.html#whatsnew-pep-448) for `*args` and `**kwargs`:

函数对于参数`*args`和`**kwargs`同样[支持](https://docs.python.org/3/whatsnew/3.5.html#whatsnew-pep-448)
```
Python 3.5+
do_something(**{**default_settings, **custom_settings})

# Also possible, this code also checks there is no intersection between keys of dictionaries
do_something(**first_args, **second_args)
```

## 具有关键字参数的面向未来的API

让我们看一下这个代码片段：
```python
model = sklearn.svm.SVC(2, 'poly', 2, 4, 0.5)
```
> Obviously, an author of this code didn't get the Python style of coding yet (most probably, just jumped from cpp or rust).
Unfortunately, this is not just question of taste, because changing the order of arguments (adding/deleting) in `SVC` will break this code. In particular, `sklearn` does some reordering/renaming from time to time of numerous algorithm parameters to provide consistent API. Each such refactoring may drive to broken code.

很明显，代码的作者还未理解Python的编码风格（很有可能是从cpp或者rust转到Python的）。
不幸的是，这不仅仅是品味的问题，因为在`SVC`中改变参数顺序（添加/删除）都会破坏代码。 特别是，`sklearn`会不时地对许多算法参数进行重新排序/重命名以提供一致的API。 每个这样的重构都可能导致代码损坏。

> In Python 3, library authors may demand explicitly named parameters by using `*`:

在Python 3中，类库作者可能会通过使用`*`来要求明确命名的参数：
```
class SVC(BaseSVC):
    def __init__(self, *, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, ... )
```
> - users have to specify names of parameters `sklearn.svm.SVC(C=2, kernel='poly', degree=2, gamma=4, coef0=0.5)` now
> - this mechanism provides a great combination of reliability and flexibility of APIs

- 用户现在必须指定参数名称为`sklearn.svm.SVC(C=2, kernel='poly', degree=2, gamma=4, coef0=0.5)`
- 这种机制提供了API完美结合的可靠性和灵活性

## 次要: `math`模块中的常量

```python
# Python 3
math.inf # 'largest' number
math.nan # not a number

max_quality = -math.inf  # no more magic initial values!

for model in trained_models:
    max_quality = max(max_quality, compute_quality(model, data))
```

## 次要: 单一的整数类型

> Python 2 provides two basic integer types, which are int (64-bit signed integer) and long for long arithmetics (quite confusing after C++).

Python 2提供了两种基础的整数类型，int(64位有符号的整数)以及对于长整型计算的long（在C++之后就变得非常混乱）。

> Python 3 has a single type `int`, which incorporates long arithmetics.

Python 3有着单一的类型`int`，其同时融合了长整型的计算。

> Here is how you check that value is integer:

如下为如何检查该值是整数：

```
isinstance(x, numbers.Integral) # Python 2, the canonical way
isinstance(x, (long, int))      # Python 2
isinstance(x, int)              # Python 3, easier to remember
```

## 其他事项

- `Enum`s are theoretically useful, but
    - string-typing is already widely adopted in the python data stack
    - `Enum`s don't seem to interplay with numpy and categorical from pandas
- coroutines also *sound* very promising for data pipelining (see [slides](http://www.dabeaz.com/coroutines/Coroutines.pdf) by David Beazley), but I don't see their adoption in the wild.
- Python 3 has [stable ABI](https://www.python.org/dev/peps/pep-0384/)
- Python 3 supports unicode identifies (so `ω = Δφ / Δt` is ok), but you'd [better use good old ASCII names](https://stackoverflow.com/a/29855176/498892)
- some libraries e.g. [jupyterhub](https://github.com/jupyterhub/jupyterhub) (jupyter in cloud), django and fresh ipython only support Python 3, so features that sound useless for you are useful for libraries you'll probably want to use once.

- `Enum`（枚举类）理论上是有用的，但是
   - string-typing 已经在Python数据栈中被广泛采用
   - `Enum`似乎不会与numpy和pandas的分类相互作用
- 协程（coroutines）*听起来*也非常适用于数据管道（参见David Beazley的[幻灯片](http://www.dabeaz.com/coroutines/Coroutines.pdf)），但是我从来没见过代码引用它们。
- Python 3 有着[稳定的ABI](https://www.python.org/dev/peps/pep-0384/)

  *ABI（Application Binary Interface）: 应用程序二进制接口 描述了应用程序和操作系统之间，一个应用和它的库之间，或者应用的组成部分之间的低接口。*
- Python 3支持unicode标识（甚至`ω=Δφ/Δt`也可以），但是你[最好使用好的旧ASCII名称](https://stackoverflow.com/a/29855176/498892)。
- 一些类库例如 [jupyterhub](https://github.com/jupyterhub/jupyterhub)（云端的jupyter），django和最新的ipython仅支持Python 3，因此对于您来说听起来无用的功能，对于您可能想要使用的库却很有用。

### 特定于数据科学的代码迁移问题（以及如何解决这些问题）

> - support for nested arguments [was dropped](https://www.python.org/dev/peps/pep-3113/)
- 对于嵌套参数的支持[已被删除](https://www.python.org/dev/peps/pep-3113/)
  ```
  map(lambda x, (y, z): x, z, dict.items())
  ```

>  However, it is still perfectly working with different comprehensions:

但是，它仍然完全适用于不同的（列表）解析：
  ```python
  {x:z for x, (y, z) in d.items()}
  ```
>  In general, comprehensions are also better 'translatable' between Python 2 and 3.

一般来说，Python 2和Python 3之间的解析也是有着更好的“可翻译性”。

> - `map()`, `.keys()`, `.values()`, `.items()`, etc. return iterators, not lists. Main problems with iterators are:
   - no trivial slicing
   - can't be iterated twice

- `map()`， `.keys()`， `.values()`， `.items()`等等返回的是迭代器（iterators），而不是列表（lists）。迭代器的主要问题如下：
   - 没有细小的切片
   - 不能迭代两次

>  Almost all of the problems are resolved by converting result to list.

将结果转换为列表几乎可以解决所有问题。

> - see [Python FAQ: How do I port to Python 3?](https://eev.ee/blog/2016/07/31/python-faq-how-do-i-port-to-python-3/) when in trouble.

- 当你遇到问题时请参见[Python FAQ: How do I port to Python 3?](https://eev.ee/blog/2016/07/31/python-faq-how-do-i-port-to-python-3/)。

### 使用python教授机器学习和数据科学的主要问题

> Course authors should spend time in the first lectures to explain what is an iterator,
why it can't be sliced / concatenated / multiplied / iterated twice like a string (and how to deal with it).

课程讲解者应该花时间在第一讲中解释什么是迭代器，
为什么它不能像字符串一样被分割/连接/相乘/重复两次（以及如何处理它）。

> I think most course authors would be happy to avoid these details, but now it is hardly possible.

我认为大多数课程讲解者曾经都乐于避开这些细节，但现在几乎不可能（再避开了）。

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


