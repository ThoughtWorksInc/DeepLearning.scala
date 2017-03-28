package com.thoughtworks

/**
  * ==Overview==
  *
  * [[com.thoughtworks.deeplearning.CumulativeLayer `CumulativeLayer`]], [[com.thoughtworks.deeplearning.DifferentiableAny `DifferentiableAny`]], [[com.thoughtworks.deeplearning.DifferentiableNothing `DifferentiableNothing`]], [[com.thoughtworks.deeplearning.Layer `Layer`]], [[com.thoughtworks.deeplearning.Poly `Poly`]] and [[com.thoughtworks.deeplearning.Symbolic `Symbolic`]] are base packages which contains necessary operations , all other packages dependent on those base packages.
  *
  * If you want to implement a layer, you need to know how to use base packages.
  *
  *
  * == Imports guidelines ==
  *
  * If you want use some operations of Type T, you should import:
  * {{{import com.thoughtworks.deeplearning.DifferentiableT._}}}
  * it means: If you want use some operations of INDArray, you should import:
  * {{{import com.thoughtworks.deeplearning.DifferentiableINDArray._}}}
  *
  * If you write something like this:
  *
  * {{{
  * def softmax(implicit scores: INDArray @Symbolic): INDArray @Symbolic = {
  *   val expScores = exp(scores)
  *   expScores / expScores.sum(1)
  * }
  * }}}
  *
  * If compiler shows error :
  * {{{ Could not infer implicit value for com.thoughtworks.deeplearning.Symbolic[org.nd4j.linalg.api.ndarray.INDArray]}}}
  * you need add import this time :
  * {{{import com.thoughtworks.deeplearning.DifferentiableINDArray._}}}
  *
  * If you write something like this:
  *
  * {{{
  * def crossEntropyLossFunction(
  *   implicit pair: (INDArray :: INDArray :: HNil) @Symbolic)
  * : Double @Symbolic = {
  *  val score = pair.head
  *  val label = pair.tail.head
  *  -(label * log(score * 0.9 + 0.1) + (1.0 - label) * log(1.0 - score * 0.9)).mean
  * }
  * }}}
  *
  * If the compiler shows error:
  * {{{value * is not a member of com.thoughtworks.deeplearning.Layer.Aux[com.thoughtworks.deeplearning.Layer.Tape.Aux[org.nd4j.linalg.api.ndarray.INDArray,org.nd4j.linalg.api.ndarray.INDArray],com.thoughtworks.deeplearning.DifferentiableINDArray.INDArrayPlaceholder.Tape]val bias = Nd4j.ones(numberOfOutputKernels).toWeight * 0.1...}}}
  * you need add import :
  * {{{
  * import com.thoughtworks.deeplearning.Poly.MathMethods.*
  * import com.thoughtworks.deeplearning.DifferentiableINDArray._
  * }}}
  *
  *
  * If the compiler shows error:
  * {{{not found: value log -(label * log(score * 0.9 + 0.1) + (1.0 - label) * log(1.0 - score * 0.9)).mean...}}}
  * you need add import:
  * {{{
  * import com.thoughtworks.deeplearning.Poly.MathFunctions.*
  * import com.thoughtworks.deeplearning.DifferentiableINDArray._
  * }}}
  *
  * Those `+` `-` `*` `/` and `log` `exp` `abs` `max` `min` are defined at [[Poly.MathMethods MathMethods]] and [[Poly.MathFunctions MathFunctions]]，those method are been implemented at DifferentiableType，so you need to import the implicit of DifferentiableType.
  *
  * == Composability ==
  *
  * Neural networks created by DeepLearning.scala are composable. You can create large networks by combining smaller networks. If two larger networks share some sub-networks, the weights in shared sub-networks trained with one network affect the other network.
  *
  * @see [[com.thoughtworks.deeplearning.DifferentiableAny.Layers.Compose Compose]]
  */
package object deeplearning {}
