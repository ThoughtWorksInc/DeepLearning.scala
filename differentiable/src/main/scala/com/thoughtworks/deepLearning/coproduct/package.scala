package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.boolean.utilities._
import com.thoughtworks.deepLearning.Batch.Aux
import com.thoughtworks.deepLearning.NeuralNetwork.Aux
import com.thoughtworks.deepLearning.Type.{DataOf, DeltaOf}
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.coproduct.ast._

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

  /** @template */
  type :+:[Head <: Type[_, _], Tail <: Coproduct] =
    Type[shapeless.:+:[DataOf[Head], DataOf[Tail]], shapeless.:+:[DeltaOf[Head], DeltaOf[Tail]]]

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
                                                             implicit headToNeuralNetwork: ToNeuralNetwork.Aux[HeadCase, Input, OutputData, OutputDelta],
                                                             tailToNeuralNetwork: ToNeuralNetwork.Aux[TailCase, Input, OutputData, OutputDelta])
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
    implicit toNeuralNetwork: ToNeuralNetwork.Aux[From, Input, OutputData, OutputDelta],
    toCoproductAst: NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< NeuralNetwork.Aux[
        Input,
        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ): CConsOps[Input, HeadData, HeadDelta, TailData, TailDelta] = {
    new CConsOps[Input, HeadData, HeadDelta, TailData, TailDelta](toCoproductAst(toNeuralNetwork(from)))
  }
}
