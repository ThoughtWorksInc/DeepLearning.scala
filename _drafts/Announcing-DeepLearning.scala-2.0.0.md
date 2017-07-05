---
title: Announcing DeepLearning.scala 2.0.0
featured: images/pic01.jpg
layout: post
---

Today, we are happy to announce DeepLearning.scala 2.0.0, the new stable release of DeepLearning.scala, a simple language for creating complex neural networks.

## Features in DeepLearning.scala 2.0

### Dynamic neural networks

Unlike some other deep learning frameworks, the structure of neural networks in DeepLearning.scala is dynamically determined during running. Our neural networks are programs. All Scala features, including functions and expressions, are available in neural networks.

For example:

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

The above neural network will go into different subnetworks according to an ordinary Scala function.

With the ability of creating dynamic neural networks, regular programmers are able to build complex neural networks from simple code. You write code almost as usual, the only difference being that code based on DeepLearning.scala is differentiable, which enables such code to evolve by modifying its parameters continuously.

### Functional programming

DeepLearning.scala 2.0 is based on Monads, which are composable, thus a complex layer can be built from primitive operators. Along with the Monad, we provide an Applicative type class, to perform multiple calculations in parallel.

For example, the previous example can be rewritten in higher-order function style as following:

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

The key construct in DeepLearning.scala 2.0 is the dependent type class [DeepLearning](https://javadoc.io/page/com.thoughtworks.deeplearning/deeplearning_2.11/latest/com/thoughtworks/deeplearning/DeepLearning.html), which witnesses a differentiable expression. In other word, by providing `DeepLearning` type class instances for these types, you can activate the deep learning ability to any types.

### Object-oriented programming

The code base of DeepLearning.scala 2.0 is organized according to Dependent Object Type calculus (DOT). All features are provided as mixin-able plugins. A plugin is able to change behaviors or APIs of all DeepLearning.scala types. This approach not only resolves [expression problem](https://en.wikipedia.org/wiki/Expression_problem), but also gives plugins the additional ability of **virtually depending** on other plugins.

### Static type system

As always, all of the above features are statically type checked.

## Plugins for DeepLearning.scala 2.0


* （此处将列出我们编写的插件，让用户感觉DeepLearning.scala 2.0功能很丰富）
* [Add your own plugin here](http://deeplearning.thoughtworks.school/get-involved)

## Links

* [DeepLearning.scala homepage](http://deeplearning.thoughtworks.school/)
* [DeepLearning.scala on Github](https://github.com/ThoughtWorksInc/DeepLearning.scala/)
* [Getting Started for DeepLearning.scala 2.0](http://deeplearning.thoughtworks.school/demo/2.0.0-Preview/GettingStarted.html)
* [Scaladoc](https://javadoc.io/page/com.thoughtworks.deeplearning/deeplearning_2.11/latest/com/thoughtworks/deeplearning/package.html)
