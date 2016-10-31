package com.thoughtworks.deepLearning
import any.Any
import hlist.ast._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object hlist {

  implicit final class HConsOps[Input <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList,
  TailDelta <: shapeless.Coproduct](
      val differentiable: Differentiable.Aux[
        Input,
        Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) {
    def head = Head(differentiable)

    def tail = Tail(differentiable)
  }

  type HList = {
    type Data <: shapeless.HList
    type Delta <: shapeless.Coproduct
  }

  type HNil = {
    type Data = shapeless.HNil
    type Delta = shapeless.CNil
  }

  type ::[Head <: Any, Tail <: HList] = {
    type Data = shapeless.::[Head#Data, Tail#Data]
    type Delta = shapeless.:+:[Head#Delta, Tail#Delta]
  }
}
