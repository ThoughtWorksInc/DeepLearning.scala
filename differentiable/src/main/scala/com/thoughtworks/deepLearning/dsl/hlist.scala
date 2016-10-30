package com.thoughtworks.deepLearning.dsl

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
sealed trait HList extends Any {
  type Data <: shapeless.HList
  type Delta <: shapeless.Coproduct
}

sealed trait HNil extends HList {
  type Data = shapeless.HNil
  type Delta = shapeless.CNil
}

sealed trait ::[Head <: Any, Tail <: HList] extends HList {
  type Data = shapeless.::[Head#Data, Tail#Data]
  type Delta = shapeless.:+:[Head#Delta, Tail#Delta]
}
