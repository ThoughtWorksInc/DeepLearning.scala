---
title: Announcing DeepLearning.scala 1.0.0
featured: images/pic01.jpg
layout: post
---

Version 1.0.0 is the first stable release of [DeepLearning.scala](http://deeplearning.thoughtworks.school/), a simple language for creating complex neural networks.

Along with the library, we created [a series of tutorials](http://deeplearning.thoughtworks.school/doc/) for developers who want to learn deep learning algorithms.

## Features in 1.0.0

### Differentiable basic types

Like [Theano](http://deeplearning.net/software/theano/) and other deep learning toolkits, DeepLearning.scala allows you to build neural networks from mathematical formulas. It supports [float](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableFloat$.html)s, [double](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableDouble$.html)s, [GPU-accelerated N-dimensional array](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableINDArray$.html)s, and calculates derivatives of the weights in the formulas.

### Differentiable ADTs

Neural networks created by DeepLearning.scala support [ADT](https://en.wikipedia.org/wiki/Algebraic_data_type) data structures (e.g. [HList](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableHList$.html) and [Coproduct](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableCoproduct$.html)), and calculate derivatives through these data structures.

### Differentiable control flow

Neural networks created by DeepLearning.scala may contains control flows like `if`/`else`/`match`/`case` in a regular language. Combined with ADT data structures, you can implement arbitary algorithms inside neural networks, and still keep some of the variables used in the algorithms differentiable and trainable.

### Composability

Neural networks created by DeepLearning.scala are composable. You can create large networks by combining smaller networks. If two larger networks share some sub-networks, the weights in shared sub-networks trained with one network affect the other network.

### Static type system

All of the above features are statically type checked.

## Links

* [Tutorials](https://thoughtworksinc.github.io/DeepLearning.scala/doc/)
* [Scaladoc](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/package.html)
* [Chat room](https://gitter.im/ThoughtWorksInc/DeepLearning.scala)

## Acknowledges

DeepLearning.scala is heavily inspired by [@MarisaKirisame](https://github.com/MarisaKirisame). Originally, we worked together for a prototype of deep learning framework, then we split our work aprt to this project and [DeepDarkFantasy](https://github.com/ThoughtWorksInc/DeepDarkFantasy).

[@milessabin](https://github.com/milessabin)'s [shapeless](https://github.com/milessabin/shapeless) provides a solid foundation for type-level programming as used in DeepLearning.scala.
