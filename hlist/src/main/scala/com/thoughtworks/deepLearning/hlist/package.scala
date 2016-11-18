package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.any.{ToNeuralNetwork, Type}
import com.thoughtworks.deepLearning.any.Type.{DataOf, DeltaOf}
import com.thoughtworks.deepLearning.hlist.ast._

import scala.language.implicitConversions
import scala.language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object hlist {

  /** @template */
  type HList = Type[_ <: shapeless.HList, _ <: shapeless.Coproduct]

  /** @template */
  type HNil = Type[shapeless.HNil, shapeless.CNil]

  /** @template */
  type ::[Head <: Type[_, _], Tail <: HList] =
    Type[shapeless.::[DataOf[Head], DataOf[Tail]], shapeless.:+:[DeltaOf[Head], DeltaOf[Tail]]]

  val HNil: ast.HNil.type = ast.HNil

  implicit def hnilToNeuralNetwork[InputData, InputDelta](implicit inputType: Type[InputData, InputDelta])
    : ToNeuralNetwork.Aux[ast.HNil.type, Batch.Aux[InputData, InputDelta], shapeless.HNil, shapeless.CNil] =
    new ToNeuralNetwork[ast.HNil.type, Batch.Aux[InputData, InputDelta]] {
      override type OutputData = shapeless.HNil
      override type OutputDelta = shapeless.CNil

      override def apply(hnil: ast.HNil.type) = hnil
    }

  final class HListOps[Input <: Batch, TailData <: shapeless.HList, TailDelta <: shapeless.Coproduct](
      tail: NeuralNetwork.Aux[Input, Batch.Aux[TailData, TailDelta]]) {

    def ::[Head, HeadData, HeadDelta](head: Head)(
        implicit headToNeuralNetwork: ToNeuralNetwork.Aux[Head, Input, HeadData, HeadDelta])
      : NeuralNetwork.Aux[Input, Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]] = {
      HCons[Input, HeadData, HeadDelta, TailData, TailDelta](headToNeuralNetwork(head), tail)
    }

  }

  implicit def toHListOps[From, Input <: Batch, TailData <: shapeless.HList, TailDelta <: shapeless.Coproduct](
      from: From)(
      implicit toNeuralNetwork: ToNeuralNetwork.Aux[From, Input, TailData, TailDelta]
  ): HListOps[Input, TailData, TailDelta] = {
    new HListOps[Input, TailData, TailDelta](toNeuralNetwork(from))
  }

  final class HConsOps[Input <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList,
  TailDelta <: shapeless.Coproduct](
      hcons: NeuralNetwork.Aux[Input,
                               Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) {
    def head: NeuralNetwork.Aux[Input, Batch.Aux[HeadData, HeadDelta]] =
      Head[Input, HeadData, HeadDelta, TailData, TailDelta](hcons)

    def tail: NeuralNetwork.Aux[Input, Batch.Aux[TailData, TailDelta]] =
      Tail[Input, HeadData, HeadDelta, TailData, TailDelta](hcons)
  }

  implicit def toHConsOps[From,
                          Input <: Batch,
                          OutputData,
                          OutputDelta,
                          HeadData,
                          HeadDelta,
                          TailData <: shapeless.HList,
                          TailDelta <: shapeless.Coproduct](from: From)(
      implicit toNeuralNetwork: ToNeuralNetwork.Aux[From, Input, OutputData, OutputDelta],
      toHListAst: NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< NeuralNetwork.Aux[
        Input,
        Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ): HConsOps[Input, HeadData, HeadDelta, TailData, TailDelta] = {
    new HConsOps[Input, HeadData, HeadDelta, TailData, TailDelta](toHListAst(toNeuralNetwork(from)))
  }

}
