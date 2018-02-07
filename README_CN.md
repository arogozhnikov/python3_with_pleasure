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

Python 不再是一种小型的脚本语言，数据管道现如今包含数个级别，而每一级又涉及到不同的框架（而且有时是非常迥异的逻辑）。

> Type hinting was introduced to help with growing complexity of programs, so machines could help with code verification.
Previously different modules used custom ways to point [types in doctrings](https://www.jetbrains.com/help/pycharm/type-hinting-in-pycharm.html#legacy)
(Hint: pycharm can convert old docstrings to fresh typehinting).



# 结论

> Python 2 and Python 3 have co-existed for almost 10 years, but we *should* move to Python 3.

虽然Python 2 和 Python 3 已经共存了十年有余，但是我们*应该*过渡到Python 3 了。

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

