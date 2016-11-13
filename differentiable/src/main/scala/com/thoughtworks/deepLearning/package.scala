package com.thoughtworks
import com.thoughtworks.deepLearning.Batch.Aux
import com.thoughtworks.deepLearning.NeuralNetwork.Aux
import com.thoughtworks.deepLearning.IsNeuralNetwork.{AstPoly1, AstPoly2}

import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object deepLearning {

  implicit final class AstOps[Left](left: Left) {

    def -[Right](right: Right)(implicit methodCase: AstMethods.-.Case[Left, Right]): methodCase.Result =
      AstMethods.-(left, right)

    def +[Right](right: Right)(implicit methodCase: AstMethods.+.Case[Left, Right]): methodCase.Result =
      AstMethods.+(left, right)

    def *[Right](right: Right)(implicit methodCase: AstMethods.*.Case[Left, Right]): methodCase.Result =
      AstMethods.*(left, right)

    def /[Right](right: Right)(implicit methodCase: AstMethods./.Case[Left, Right]): methodCase.Result =
      AstMethods./(left, right)

  }

  implicit def autoToLiteral[A, Input <: Batch, OutputData, OutputDelta](a: A)(
      implicit isNeuralNetwork: IsNeuralNetwork.Aux[A, Input, OutputData, OutputDelta]): NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
    isNeuralNetwork(a)
  }

  implicit final class ToLiteralOps[A](a: A) {
    def toLiteral[Input <: Batch, OutputData, OutputDelta](
        implicit isNeuralNetwork: IsNeuralNetwork.Aux[A, Input, OutputData, OutputDelta]): NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
      isNeuralNetwork(a)
    }
  }

  object log extends AstPoly1
  object exp extends AstPoly1
  object abs extends AstPoly1
  object max extends AstPoly2
  object min extends AstPoly2

}

package deepLearning {

  object AstMethods {
    object - extends AstPoly2
    object + extends AstPoly2
    object * extends AstPoly2
    object / extends AstPoly2
  }

}
