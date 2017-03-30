# DeepLearning.scala  <a href="http://thoughtworks.com/"><img align="right" src="https://www.thoughtworks.com/imgs/tw-logo.png" title="ThoughtWorks" height="15"/></a>

[![Join the chat at https://gitter.im/ThoughtWorksInc/DeepLearning.scala](https://badges.gitter.im/ThoughtWorksInc/DeepLearning.scala.svg)](https://gitter.im/ThoughtWorksInc/DeepLearning.scala?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/ThoughtWorksInc/DeepLearning.scala.svg)](https://travis-ci.org/ThoughtWorksInc/DeepLearning.scala)
[![Scaladoc](https://javadoc.io/badge/com.thoughtworks.deeplearning/unidoc_2.11.svg?label=scaladoc)](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/package.html)

[![DifferentiableAny](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiableany/latest.svg)](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiableany)
[![DifferentiableNothing](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiablenothing/latest.svg)](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiablenothing)
[![DifferentiableSeq](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiableseq/latest.svg)](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiableseq)
[![DifferentiableDouble](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiabledouble/latest.svg)](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiabledouble)
[![DifferentiableFloat](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiablefloat/latest.svg)](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiablefloat)
[![DifferentiableHList](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiablehlist/latest.svg)](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiablehlist)
[![DifferentiableCoproduct](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiablecoproduct/latest.svg)](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiablecoproduct)
[![DifferentiableINDArray](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiableindarray/latest.svg)](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/differentiableindarray)

**DeepLearning.scala** is a DSL for creating complex neural networks.

With the help of DeepLearning.scala, regular programmers are able to build complex neural networks from simple code. You write code almost as usual, the only difference being that code based on DeepLearning.scala is [differentiable](https://colah.github.io/posts/2015-09-NN-Types-FP/), which enables such code to evolve by modifying its parameters continuously.

## Features

### Differentiable basic types

Like [Theano](http://deeplearning.net/software/theano/) and other deep learning toolkits, DeepLearning.scala allows you to build neural networks from mathematical formulas. It supports [floats](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableFloat$.html), [doubles](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableDouble$.html), [GPU-accelerated N-dimensional arrays](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableINDArray$.html), and calculates derivatives of the weights in the formulas.

### Differentiable ADTs

Neural networks created by DeepLearning.scala support [ADT](https://en.wikipedia.org/wiki/Algebraic_data_type) data structures (e.g. [HList](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableHList$.html) and [Coproduct](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableCoproduct$.html)), and calculate derivatives using these data structures.

### Differentiable control flow

Neural networks created by DeepLearning.scala may contain control flows like `if`/`else`/`match`/`case` in a regular language. Combined with ADT data structures, you can implement arbitrary algorithms inside neural networks, and still keep the variables within those algorithms differentiable and trainable.

### Composability

Neural networks created by DeepLearning.scala are composable. You can create large networks by combining smaller networks. If two larger networks share sub-networks, the weights for a shared sub-network that are trained in one super-network will also affect the other super-network.

### Static type system

All of the above features are statically type checked.

## Roadmap

### v1.0

Version 1.0 is the current version with all of the above features. The final version will be released in March 2017.

### v2.0

* Support `for`/`while` and other higher-order functions on differentiable `Seq`s.
* Support `for`/`while` and other higher-order functions on GPU-accelerated differentiable N-dimensional arrays.

Version 2.0 will be released in Q2 2017.

### v3.0

* Support using custom `case class`es inside neural networks.
* Support distributed models and distributed training on [Spark](https://spark.apache.org/).

Version 3.0 will be released in late 2017.

## Links

* [Getting started](https://thoughtworksinc.github.io/DeepLearning.scala/demo/GettingStarted.html)
* [Scaladoc](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/package.html)

## Acknowledgements

DeepLearning.scala is heavily inspired by my colleague [@MarisaKirisame](https://github.com/MarisaKirisame). Originally, we worked together on a prototype of a deep learning framework, and eventually split our work into this project and [DeepDarkFantasy](https://github.com/ThoughtWorksInc/DeepDarkFantasy).

[@milessabin](https://github.com/milessabin)'s [shapeless](https://github.com/milessabin/shapeless) provides a solid foundation for type-level programming as used in DeepLearning.scala.
