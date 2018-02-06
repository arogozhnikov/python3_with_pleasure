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
