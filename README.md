# <a href="http://deeplearning.thoughtworks.school/"><img src="http://deeplearning.thoughtworks.school/assets/images/logo-text-black.png" alt="DeepLearning.scala" height="40"/></a>  <a href="http://thoughtworks.com/"><img align="right" src="https://www.thoughtworks.com/imgs/tw-logo.png" alt="ThoughtWorks" height="15"/></a>

[![Join the chat at https://gitter.im/ThoughtWorksInc/DeepLearning.scala](https://badges.gitter.im/ThoughtWorksInc/DeepLearning.scala.svg)](https://gitter.im/ThoughtWorksInc/DeepLearning.scala?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/ThoughtWorksInc/DeepLearning.scala.svg?branch=3.0.x)](https://travis-ci.org/ThoughtWorksInc/DeepLearning.scala)
[![Latest version](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/plugins-builtins/latest.svg)](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/plugins-builtins)
[![Scaladoc](https://javadoc.io/badge/com.thoughtworks.deeplearning/deeplearning_2.11.svg?label=scaladoc)](https://javadoc.io/page/com.thoughtworks.deeplearning/deeplearning_2.11/latest/com/thoughtworks/deeplearning/package.html)

**DeepLearning.scala** is a simple library for creating complex neural networks from object-oriented and functional programming constructs.
 
 * DeepLearning.scala runs on JVM, can be used either in standalone JVM applications or a Jupyter Notebooks.
 * DeepLearning.scala is expressive. Various types of neural network layers can be created by composing `map`, `reduce` or other higher order functions.
 * DeepLearning.scala supports plugins. There are various plugins providing algorithms, models, hyperparameters or other features.
 * All the above features are statically type checked.

## Features

### Differentiable programming

Like other deep learning toolkits, DeepLearning.scala allows you to build neural networks from mathematical formulas. It supports [floats](https://javadoc.io/page/com.thoughtworks.deeplearning/deeplearning_2.11/latest/com/thoughtworks/deeplearning/plugins/FloatLayers.html), [doubles](https://javadoc.io/page/com.thoughtworks.deeplearning/deeplearning_2.11/latest/com/thoughtworks/deeplearning/plugins/DoubleLayers.html), [GPU-accelerated N-dimensional arrays](https://javadoc.io/page/com.thoughtworks.deeplearning/deeplearning_2.11/latest/com/thoughtworks/deeplearning/plugins/INDArrayLayers.html), and calculates derivatives of the weights in the formulas.

### Dynamic neural networks

Unlike some other deep learning toolkits, the structure of neural networks in DeepLearning.scala is dynamically determined during running. Our neural networks are programs. All Scala features, including functions, expressions and control flows, are available in neural networks.

For example:

``` scala
def ordinaryScalaFunction(a: INDArray): Boolean = {
  a.signnum.sumT > math.random
}

def myDynamicNeuralNetwork(input: INDArray) = INDArrayLayer(monadic[Do] {
  val outputOfLayer1 = layer1(input).forward.each
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

DeepLearning.scala 2.0 is based on Monads, which are composable, thus a complex layer can be built from primitive operators or higher order functions like `map`/`reduce`. Along with the Monad, we provide an Applicative type class, to perform multiple calculations in parallel.

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

The key construct in DeepLearning.scala 2.0 is the dependent type class [DeepLearning](https://javadoc.io/page/com.thoughtworks.deeplearning/deeplearning_2.11/latest/com/thoughtworks/deeplearning/DeepLearning.html), which witnesses a differentiable expression. In other words, given the `DeepLearning` type class instance, you can activate the deep learning ability of any type.

### Object-oriented programming

The code base of DeepLearning.scala 2.0 is organized according to Dependent Object Type calculus (DOT). All features are provided as mixin-able plugins. A plugin is able to change APIs and behaviors of all DeepLearning.scala types. This approach not only resolves [expression problem](https://en.wikipedia.org/wiki/Expression_problem), but also gives plugins the additional ability of **virtually depending** on other plugins.

For example, when a plugin author is creating the [Adagrad](https://gist.github.com/Atry/89ee1baa4c161b8ccc1b82cdd9c109fe#file-adagrad-sc) optimizer plugin, he does not have to explicitly call functions related to learning rate. However, once a plugin user enables both the `Adagrad` plugin and the [FixedLearningRate](https://gist.github.com/Atry/1fb0608c655e3233e68b27ba99515f16#file-readme-ipynb) plugin, then computation in `FixedLearningRate` will get called eventually when the `Adagrad` optimization is executed.

## Roadmap

### v2.0

Version 2.0 is the current version with all of the above features.

### v3.0

* Support element-wise `map`/`reduce` and other higher-order functions on GPU.
* Support distributed models and distributed training on [Spark](https://spark.apache.org/).

Version 3.0 will be released in late 2017.

## Links

* [Homepage](http://deeplearning.thoughtworks.school/)
* [Getting started](https://thoughtworksinc.github.io/DeepLearning.scala/demo/GettingStarted.html)
* [Scaladoc](https://javadoc.io/page/com.thoughtworks.deeplearning/deeplearning_2.11/latest/com/thoughtworks/deeplearning/package.html)

## Acknowledgements

DeepLearning.scala is sponsored by [ThoughtWorks](https://www.thoughtworks.com/).

DeepLearning.scala is heavily inspired by my colleague [@MarisaKirisame](https://github.com/MarisaKirisame). Originally, we worked together on a prototype of a deep learning framework, and eventually split our work into this project and [DeepDarkFantasy](https://github.com/ThoughtWorksInc/DeepDarkFantasy).
Other contributors can be found at [here](https://github.com/ThoughtWorksInc/DeepLearning.scala/graphs/contributors).

### Related projects

 * [Shapeless](https://github.com/milessabin/shapeless) provides a solid foundation for type-level programming used in DeepLearning.scala.
 * [Scalaz](http://scalaz.org/) and [Algebra](http://typelevel.org/algebra/) provides type classes used in DeepLearning.scala.
 * [ThoughtWorks Each](https://github.com/ThoughtWorksInc/each) provides `async`/`await`-like syntax. You may want to use it to control your training process in an imperative style.
 * [nd4j](http://nd4j.org/) provides numerical computing used in DeepLearning.scala.
 * [RAII.scala](https://github.com/ThoughtWorksInc/RAII.scala), [future.scala](https://github.com/ThoughtWorksInc/future.scala) and [tryt.scala](https://github.com/ThoughtWorksInc/tryt.scala) provides monadic asynchronous resource management used in DeepLearning.scala.
 * Plugins of DeepLearning.scala are based on [Factory](https://javadoc.io/page/com.thoughtworks.feature/factory_2.11/latest/com/thoughtworks/feature/Factory.html) and other dependent type classes in [feature.scala](https://github.com/ThoughtWorksInc/feature.scala).
 * [Import.scala](https://github.com/ThoughtWorksInc/Import.scala) is a Scala compiler plugin that enables magic imports. You may need it in those sbt project use DeepLearning.scala plugin hosted on Github Gist.
 * DeepLearning.scala can run in [Jupyter Scala](https://github.com/alexarchambault/jupyter-scala) or [Ammonite](http://ammonite.io/).
 * The unit tests of DeepLearning.scala are written in [ScalaTest](http://scalatest.org/) and [example.scala](https://javadoc.io/page/com.thoughtworks.example/unidoc_2.12/latest/com/thoughtworks/example.html) syntax.
 * Some type classes in DeepLearning.scala are created by [simulacrum](https://github.com/mpilquist/simulacrum)'s `@typeclass` annotation.
