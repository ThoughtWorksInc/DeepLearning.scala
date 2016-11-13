package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.boolean.utilities._
import com.thoughtworks.deepLearning.Batch.Aux
import com.thoughtworks.deepLearning.NeuralNetwork.Aux
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.coproduct.ast._
import shapeless.Lazy
import shapeless.ops.coproduct.IsCCons

import scala.language.existentials
import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object coproduct {

  /** @template */
  type Coproduct = Type[_ <: shapeless.Coproduct, _ <: shapeless.Coproduct]

  /** @template */
  type CNil = Type[shapeless.CNil, shapeless.CNil]
  val CNil: CNil = implicitly

  /** @template */
  type :+:[Head <: Type[_, _], Tail <: Coproduct] =
    Type[shapeless.:+:[head.Data, tail.Data], shapeless.:+:[head.Delta, tail.Delta]] forSome {
      val head: Head
      val tail: Tail
    }

  implicit final class RichCoproductType[TailData <: shapeless.Coproduct, TailDelta <: shapeless.Coproduct](
      tail: Type[TailData, TailDelta]) {
    def :+:[HeadData, HeadDelta](head: Type[HeadData, HeadDelta]) =
      new Type[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]
  }

  final class CConsOps[
      Input <: Batch,
      HeadData,
      HeadDelta,
      TailData <: shapeless.Coproduct,
      TailDelta <: shapeless.Coproduct
  ](
      ccons: NeuralNetwork.Aux[
        Input,
        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]
      ]
  ) {

    def head: NeuralNetwork.Aux[Input, Type[HeadData, HeadDelta]#Batch] =
      Head[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

    def tail: NeuralNetwork.Aux[Input, Type[TailData, TailDelta]#Batch] =
      Tail[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

    def choice[HeadCase, TailCase, OutputData, OutputDelta](
        caseHead: NeuralNetwork.Aux[Input, Type[HeadData, HeadDelta]#Batch] => HeadCase)(
        caseTail: NeuralNetwork.Aux[Input, Type[TailData, TailDelta]#Batch] => TailCase)(
                                                             implicit headIsNeuralNetwork: IsNeuralNetwork.Aux[HeadCase, Input, OutputData, OutputDelta],
                                                             tailIsNeuralNetwork: IsNeuralNetwork.Aux[TailCase, Input, OutputData, OutputDelta])
      : NeuralNetwork.Aux[Input, Type[OutputData, OutputDelta]#Batch] = {
      If[Input, Batch.Aux[OutputData, OutputDelta]](isInl, caseHead(head), caseTail(tail))
    }

    def isInl: NeuralNetwork.Aux[Input, Boolean#Batch] = IsInl[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

  }

  implicit def toCConsOps[From,
                          Input <: Batch,
                          OutputData,
                          OutputDelta,
                          HeadData,
                          HeadDelta,
                          TailData <: shapeless.Coproduct,
                          TailDelta <: shapeless.Coproduct](from: From)(
    implicit isNeuralNetwork: IsNeuralNetwork.Aux[From, Input, OutputData, OutputDelta],
    toCoproductAst: NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< NeuralNetwork.Aux[
        Input,
        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ): CConsOps[Input, HeadData, HeadDelta, TailData, TailDelta] = {
    new CConsOps[Input, HeadData, HeadDelta, TailData, TailDelta](toCoproductAst(isNeuralNetwork(from)))
  }
}
