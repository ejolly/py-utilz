[![Build Status](https://api.travis-ci.com/ejolly/utilz.svg?branch=master)](https://travis-ci.com/ejolly/utilz)
[![Coverage Status](https://coveralls.io/repos/github/ejolly/utilz/badge.svg?branch=master)](https://coveralls.io/github/ejolly/utilz?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)

# Utilz

A python package that combines several key ideas to make data analysis faster, easier, and more reliable:  

1. Defensive data analysis through the likes of [bulwark](https://bulwark.readthedocs.io/en/latest/index.html)
2. Minimization of boilerplate code, e.g. for plotting in `matplotlib` and `seaborn`
3. Common data operations in pandas, e.g. normalizing by group
4. Common i/o operations like managing paths
5. More concise and reusable code via functional programming ideas via [toolz](https://toolz.readthedocs.io/en/latest/), e.g. `>>o>>` as a pipe operator