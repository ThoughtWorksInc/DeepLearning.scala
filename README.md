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

**DeepLearning.scala** is a DSL for creating very complex neural networks.

With the help of DeepLearning.scala, a normal programmer is able to build very complex neural networks from very simple code. A programmer still writes code as usual. The only difference is that the code with DeepLearning.scala are [differentiable](https://colah.github.io/posts/2015-09-NN-Types-FP/), which let the code evolve itself and modify its parameters continuously.

## Features

### Differentiable basic types

Like [Theano](http://deeplearning.net/software/theano/) or other deep learning toolkits, DeepLearning.scala allows you to build neural networks from mathematical formula, which handle [float](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableFloat$.html)s, [double](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableDouble$.html)s and [GPU-accelerated N-dimensional array](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableINDArray$.html)s, and calculate derivatives of the weights in the formula.

### Differentiable ADT

Neural networks created by DeepLearning.scala are able to handle [ADT](https://en.wikipedia.org/wiki/Algebraic_data_type) data structures(e.g. [HList](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableHList$.html) and [Coproduct](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableCoproduct$.html)), and calculate derivatives through these data structure.

### Differentiable control flow

Neural networks created by DeepLearning.scala may contains control flows like `if`/`else`/`switch`/`case`. Combining with ADT data structures, You can implement classic mostly-non-machine-learing algorithms and train some of the variables used in the algorithms.

### Composibility

Neural networks created by DeepLearning.scala are composible. You can creates many small networks, and compose them into larger networks. If two larger networks shares some sub-networks, the weights trained with one network affects the other network.

### Static type system

All the above features are statically type checked.

## Roadmap

### v1.0

Version 1.0 is the current version with all above features. The final version will be released in Janary 2017.

### v2.0

* Support `for`/`while` and other higher-order functions on differenitable `Seq`s.
* Support `for`/`while` and other higher-order functions on GPU-accelerated differenitable N-dimensional arrays.

Version 2.0 will be released in March 2017.

### v3.0

* Support using custom `case class`es inside neural networks.
* Support distributed models and distributed training on [Spark](https://spark.apache.org/).

Version 3.0 will be released in late 2017.

## Links

* [Getting started](https://github.com/ThoughtWorksInc/DeepLearning.scala/wiki/Getting-Started)
* [Tutorial](https://github.com/ThoughtWorksInc/DeepLearning.scala/wiki/Home)
* [Scaladoc](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/package.html)

## Acknowledges

DeepLearning.scala is heavily inspired by my colleague [@MarisaKirisame](https://github.com/MarisaKirisame).

[@milessabin](https://github.com/milessabin)'s [shapeless](https://github.com/milessabin/shapeless) provides a solid foundation for type-level programming used in DeepLearning.scala.
