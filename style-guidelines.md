---
layout: markdown
title: Style Guidelines
---

## 循环依赖

所有package之间不应该有循环依赖、所有Scala文件之间不应该有循环依赖。可以用[acyclic](https://github.com/lihaoyi/acyclic)检查循环依赖。

## 库的粒度

每个库应该尽量小，只应包含一个Scala源文件。库名就是文件名（但不包含`.scala`后缀）。

## 避免package object

尽量不使用`package object`。应当使用普通object来模拟package。这种情况属于[模拟package](http://docs.scala-lang.org/style/naming-conventions.html#objects)，命名应该小写。

## 类型别名的伴生对象

类型别名应当与其伴生对象同名且位于同一文件。这个文件通常是模拟package的object。
