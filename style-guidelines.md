---
layout: markdown
title: Style Guidelines
---

## 循环依赖

package之间不应该有循环依赖。Scala文件之间不应该有循环依赖。可以用[acyclic](https://github.com/lihaoyi/acyclic)检查循环依赖。

## 库的粒度

每个库应该尽量小，只应包含一个Scala源文件。

库名通常是去掉`.scala`后缀的文件名。但有一个例外：如果文件名是`package.scala`，库名应该是包名。

## 模拟package

如果需要提供一个由一组静态成员组成的库，那么这个库不应该是`package object`，而应当使用普通`object`来模拟`package`，命名应该小写。

## package object

如果有若干个相关的库位于同一个package，可以为这一组库提供一个`package object`作为[外观模式](https://zh.wikipedia.org/wiki/%E5%A4%96%E8%A7%80%E6%A8%A1%E5%BC%8F)。这个`package object`的文件名应该是`package.scala`，库名是包名。

## 类型别名的伴生对象

类型别名应当与其伴生对象同名且位于同一文件。这个文件可以是模拟package的object。例如：

``` scala
object tryT {
  type TryT[F[_], A] = ???
  object TryT {
    ???
  }
}
```
