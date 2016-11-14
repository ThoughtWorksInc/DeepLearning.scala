package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.boolean.utilities._
import com.thoughtworks.deepLearning.Batch.Aux
import com.thoughtworks.deepLearning.NeuralNetwork.Aux
import com.thoughtworks.deepLearning.Type.{DataOf, DeltaOf}
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.coproduct.ast._
import shapeless.Lub

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

    def head: NeuralNetwork.Aux[Input, Batch.Aux[HeadData, HeadDelta]] =
      Head[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

    def tail: NeuralNetwork.Aux[Input, Batch.Aux[TailData, TailDelta]] =
      Tail[Input, HeadData, HeadDelta, TailData, TailDelta](ccons)

    def choice[HeadCase,
               TailCase,
               HeadOutputData,
               HeadOutputDelta,
               TailOutputData,
               TailOutputDelta,
               NN,
               OutputData,
               OutputDelta](caseHead: NeuralNetwork.Aux[Input, Batch.Aux[HeadData, HeadDelta]] => HeadCase)(
        caseTail: NeuralNetwork.Aux[Input, Batch.Aux[TailData, TailDelta]] => TailCase)(
        implicit headToNeuralNetwork: ToNeuralNetwork.Aux[HeadCase, Input, HeadOutputData, HeadOutputDelta],
        tailToNeuralNetwork: ToNeuralNetwork.Aux[TailCase, Input, TailOutputData, TailOutputDelta],
        lub: Lub[NeuralNetwork.Aux[Input, Batch.Aux[HeadOutputData, HeadOutputDelta]],
                 NeuralNetwork.Aux[Input, Batch.Aux[TailOutputData, TailOutputDelta]],
                 NN],
        commonToNeuralNetwork: ToNeuralNetwork.Aux[NN, Input, OutputData, OutputDelta]
    ): NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
      If[Input, Batch.Aux[OutputData, OutputDelta]](
        isInl,
        commonToNeuralNetwork(lub.left(headToNeuralNetwork(caseHead(head)))),
        commonToNeuralNetwork(lub.right(tailToNeuralNetwork(caseTail(tail)))))
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
      toCoproductNeuralNetwork: NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< NeuralNetwork.Aux[
        Input,
        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ): CConsOps[Input, HeadData, HeadDelta, TailData, TailDelta] = {
    new CConsOps[Input, HeadData, HeadDelta, TailData, TailDelta](toCoproductNeuralNetwork(toNeuralNetwork(from)))
  }
}
