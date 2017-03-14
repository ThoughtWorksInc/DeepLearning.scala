package com.thoughtworks

/**
  * This is the documentation for the DeepLearning.Scala
  *
  *  ==Overview==
  *
  * [[com.thoughtworks.deeplearning.BufferedLayer `BufferedLayer`]],[[com.thoughtworks.deeplearning.DifferentiableAny `DifferentiableAny`]],[[com.thoughtworks.deeplearning.DifferentiableNothing `DifferentiableNothing`]],[[com.thoughtworks.deeplearning.Layer `Layer`]],[[com.thoughtworks.deeplearning.Poly `Poly`]] and [[com.thoughtworks.deeplearning.Symbolic `Symbolic`]] are base packages which contains necessary ops , all other packages dependent on those base packages，
  *
  * If you want to implement a layer, you need to know how to use base packages.
  *
  *
  * == Imports guidelines ==
  *
  * If you want use some ops of Type T,you should import `import com.thoughtworks.deeplearning.DifferentiableT._` first,it means:
  * If you want use some ops of INDArray,you should import `import com.thoughtworks.deeplearning.DifferentiableINDArray._` first.
  *
  * {{{
  * def softmax(implicit scores: INDArray @Symbolic): INDArray @Symbolic = {
  *   val expScores = exp(scores)
  *   expScores / expScores.sum(1)
  * }
  * }}}
  *
  * If the compiler shows error : `Could not infer implicit value for com.thoughtworks.deeplearning.Symbolic[org.nd4j.linalg.api.ndarray.INDArray]...` ,you need add import this time :`import com.thoughtworks.deeplearning.DifferentiableINDArray._`
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
  * If the compiler shows error :` value * is not a member of com.thoughtworks.deeplearning.Layer.Aux[com.thoughtworks.deeplearning.Layer.Batch.Aux[org.nd4j.linalg.api.ndarray.INDArray,org.nd4j.linalg.api.ndarray.INDArray],com.thoughtworks.deeplearning.DifferentiableINDArray.INDArrayPlaceholder.Batch]val bias = Nd4j.ones(numberOfOutputKernels).toWeight * 0.1...`
  * you need add import this time :`import com.thoughtworks.deeplearning.Poly.MathMethods.*` and `import com.thoughtworks.deeplearning.DifferentiableINDArray._`；
  *
  *
  * If the compiler shows error :` not found: value log -(label * log(score * 0.9 + 0.1) + (1.0 - label) * log(1.0 - score * 0.9)).mean...`
  * you need add import this time :`import com.thoughtworks.deeplearning.Poly.MathFunctions.*` and `import com.thoughtworks.deeplearning.DifferentiableINDArray._`；
  *
  *
  * Those `+ - * /` and `log exp abs max min` are defined at `MathMethods` and `MathFunctions` ，the method are been implemented at DifferentiableType，so you need to import the implicit of DifferentiableType
  *
  */
package object deeplearning {}
