---
title: Announcing DeepLearning.scala 2.0.0-RC1
featured: images/pic01.jpg
layout: post
---

Today, we are happy to announce DeepLearning.scala 2.0.0-RC1, a release candidate of DeepLearning.scala 2.

DeepLearning.scala 2.0 comes with two major features in addition to DeepLearning.scala 1.0: dynamic neural network and [Factory](https://javadoc.io/page/com.thoughtworks.feature/factory_2.11/latest/com/thoughtworks/feature/Factory.html)-based plugins.

In DeepLearning.scala 2.0, a neural network is an ordinary Scala function that returns a [Layer](https://javadoc.io/page/com.thoughtworks.deeplearning/plugins-builtins_2.11/latest/com/thoughtworks/deeplearning/plugins/Layers$Layer.html), which represents the process that dynamically creates computational graph nodes, instead of static computational graphs in TensorFlow or some other deep learning frameworks. All Scala features, including functions and expressions, are available in DeepLearning.scala's dynamic neural networks.

`Factory`-based plugins resolve [expression problem](https://en.wikipedia.org/wiki/Expression_problem). Any hyperparameters, neural network optimization algorithms or special subnetworks are reusable in the simple `Factory[YouPlugin1 with YouPlugin2]` mechanism.

See [Getting Started](http://deeplearning.thoughtworks.school/demo/2.0.0-Preview/GettingStarted.html) to have a try.

---

### Links

* [DeepLearning.scala homepage](http://deeplearning.thoughtworks.school/)
* [DeepLearning.scala on Github](https://github.com/ThoughtWorksInc/DeepLearning.scala/)
* [Getting Started for DeepLearning.scala 2.0](http://deeplearning.thoughtworks.school/demo/2.0.0-Preview/GettingStarted.html)
* [Scaladoc](https://javadoc.io/page/com.thoughtworks.deeplearning/deeplearning_2.11/latest/com/thoughtworks/deeplearning/package.html)
* [Factory](https://javadoc.io/page/com.thoughtworks.feature/factory_2.11/latest/com/thoughtworks/feature/Factory.html)
