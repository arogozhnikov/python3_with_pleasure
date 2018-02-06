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


