---
title: DeepLearning.scala 2.0.0发布
featured: images/pic01.jpg
layout: post
---

今天，我们很荣幸宣布，DeepLearning.scala 2.0.0发布了。这是DeepLearning.scala的最新稳定版本。

DeepLearning.scala是个Scala库，能简简单单的创建复杂神经网络。

## DeepLearning.scala 2.0的特性

### 动态神经网络

与其他一些深度学习框架不同，DeepLearning.scala中的神经网络结构要在运行时才动态确定。我们的神经网络都是程序。所有Scala特性，包括函数和表达式，都能在神经网络中直接调用。

比如：

``` scala
def ordinaryScalaFunction(a: INDArray): Boolean = {
  a.signnum.sumT > math.random
}

def myDynamicNeuralNetwork(input: INDArray) = INDArrayLayer(monadic[Do] {
  val outputOfLayer1: INDArray = layer1(input).forward.each
  if (ordinaryScalaFunction(outputOfLayer1.data)) {
    dynamicallySelectedLayer2(outputOfLayer1).forward.each
  } else {
    dynamicallySelectedLayer3(outputOfLayer1).forward.each
  }
})
```

以上神经网络会根据`ordinaryScalaFunction`的返回值进入不同的子网络，而`ordinaryScalaFunction`只是个普通的Scala函数。


有了动态创建神经网络的能力，普通的程序员能够用很简单的代码构建复杂神经网络。你还是像以前一样写程序，唯一的区别是，DeepLearning.scala里写的程序有学习能力，能够持续根据反馈修改自身参数。

### 函数式编程

DeepLearning.scala 2.0基于Monads，所以可以任意组合。即使是很复杂的网络也可以从原子操作组合出来。除了Monad以外，我们还提供了Applicative类型类（type class），能并行执行多处耗时计算。

比如，先前的例子可以用高阶函数风格写成这样：

``` scala
def myDynamicNeuralNetwork(input: INDArray) = INDArrayLayer {
  layer1(input).forward.flatMap { outputOfLayer1 =>
    if (ordinaryScalaFunction(outputOfLayer1.data)) {
      dynamicallySelectedLayer2(outputOfLayer1).forward
    } else {
      dynamicallySelectedLayer3(outputOfLayer1).forward
    }
  }
}
```

DeepLearning.scala 2.0的核心概念是依赖类型类（dependent type class）[DeepLearning](https://javadoc.io/page/com.thoughtworks.deeplearning/deeplearning_2.11/latest/com/thoughtworks/deeplearning/DeepLearning.html)，可以见证（witness）可微分表达式。换句话说，对于任何数据类型，包括你定制的类型，只要提供了对应的`DeepLearning`类型类的实例，就能具备深度学习能力，成为深度神经网络的一部分。

### 面向对象编程

DeepLearning 2.0的代码结构利用了依赖对象类型演算（Dependent Object Type calculus，DOT）。每一个特性都是一个支持混入（mixin）的插件。所有DeepLearning.scala类型的行为和API都可以通过插件修改。这种方式不光解决了[expression problem](https://en.wikipedia.org/wiki/Expression_problem)，还让每个插件都可以“虚依赖”其他插件。

### 静态类型系统

与DeepLearning 1.0一样，DeepLearning.scala 2.0所有特性都支持静态类型检查。

## DeepLearning.scala 2.0的插件

* （此处将列出我们编写的插件，让用户感觉DeepLearning.scala 2.0功能很丰富）
* [贡献你自己的插件](http://deeplearning.thoughtworks.school/get-involved)

## Links

* [DeepLearning.scala主页](http://deeplearning.thoughtworks.school/)
* [DeepLearning.scala Github页面](https://github.com/ThoughtWorksInc/DeepLearning.scala/)
* [DeepLearning.scala 2.0快速上手指南](http://deeplearning.thoughtworks.school/demo/2.0.0-Preview/GettingStarted.html)
* [API参考文档](https://javadoc.io/page/com.thoughtworks.deeplearning/deeplearning_2.11/latest/com/thoughtworks/deeplearning/package.html)