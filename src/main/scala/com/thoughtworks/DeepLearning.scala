package com.thoughtworks

import cats.{Eval, Monoid}
import shapeless._
import simulacrum.typeclass
import scala.language.implicitConversions

/*

神经网络应该是 trait 还是 one case class 还是 witnesswith + case class 还是 xxxops + case class?

* trait
  * 优点
    * 实现起来更简单
    * 类型参数短
  * 缺点
    * 需要定义 self type 才能 applyPatch
    * 类型推断可能很难(现在由于砍掉了 SKI,可能并不难)
    * weight delta 的 monoid不好重用。
* case class
  * 优点
    * 自带 self type
    * 可以类型推断
  * 缺点
    * 类型参数很复杂

暂定用trait,尝试一下简单的解法
怎么表示input和output?
要么也用 trait ,要么还是用 type class
对于常数函数（即K/liftXxx）,需要专门实现。
对于平常的 input ,似乎并不需要type class就能传递回去

backward阶段并不需要计算 input delta


不同类型的helper应该放在神经网络上还是全局函数上?应该放在type class里。

 */
object DeepLearning {

  @typeclass
  trait Patch[Weight] {
    type WeightDelta

    def applyPatch(delta: WeightDelta): Weight
  }

  object Patch {
    type Aux[Weight, WeightDelta0] = Patch[Weight] {
      type WeightDelta = WeightDelta0
    }
  }

  object CachedForward {

    final class ForwardKeyValue[-K, +V]()

    object ForwardKeyValue {
      implicit def apply[Weight, WeightDelta, InputData, OutputData, OutputDelta] = {
        new ForwardKeyValue[((HMap[ForwardKeyValue], Weight, InputData) => (HMap[ForwardKeyValue], OutputData, OutputDelta => WeightDelta), Weight, InputData), (OutputData, OutputDelta => WeightDelta)]
      }
    }

    type ForwardMap = HMap[ForwardKeyValue]
  }

  import CachedForward._

  final case class CachedForward[Weight, WeightDelta, InputData, OutputData, OutputDelta]
  (
    rawForward: (HMap[ForwardKeyValue], Weight, InputData) => (HMap[ForwardKeyValue], OutputData, OutputDelta => WeightDelta)
  ) extends AnyVal {
    def apply(cache: HMap[ForwardKeyValue], weight: Weight, inputData: InputData): (HMap[ForwardKeyValue], OutputData, OutputDelta => WeightDelta) = {
      cache.get((rawForward, weight, inputData)) match {
        case None =>
          // Note that recursion or re-entering is not supported here
          val (c, outputData, backward) = rawForward(cache, weight, inputData)
          val newCache = c + ((rawForward, weight, inputData) -> (outputData, backward))
          (newCache, outputData, backward)
        case Some((outputData, backward: (OutputDelta => WeightDelta))) =>
          (cache, outputData, backward)
      }
    }
  }

  trait Differentiable[Weight] {
    self =>
    type WeightDelta
    type InputData
    type OutputData
    type OutputDelta

    val weight: Weight

    val monoid: Monoid[WeightDelta]

    val patch: Patch.Aux[Weight, WeightDelta]

    val forward: CachedForward[Weight, WeightDelta, InputData, OutputData, OutputDelta]
  }

  object Differentiable {
    type Aux[Weight, WeightDelta0, InputData0, OutputData0, OutputDelta0] = Differentiable[Weight] {
      type WeightDelta = WeightDelta0
      type InputData = InputData0
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
    }
  }

  final case class DifferentiableDouble[Weight, WeightDelta0, InputData0]
  (
    weight: Weight,
    monoid: Monoid[WeightDelta0],
    patch: Patch.Aux[Weight, WeightDelta0],
    forward: CachedForward[Weight, WeightDelta0, InputData0, Eval[Double], Eval[Double]]
  ) extends Differentiable[Weight] with Dsl.Double {
    override type WeightDelta = WeightDelta0
    override type InputData = InputData0
    override type OutputData = Eval[scala.Double]
    override type OutputDelta = Eval[scala.Double]
    override protected type Self = DifferentiableDouble[_, _, InputData]

    override def unary_- = {
      val outerForward = forward
      DifferentiableDouble(weight, monoid, patch, CachedForward { (forwardCache: HMap[ForwardKeyValue], weight: Weight, inputData: InputData) =>
        val (newCache, data, backward) = outerForward(forwardCache, weight, inputData)

        (
          newCache,
          data.map(-_).memoize, { outputDelta: Eval[Double] =>
          // TODO: backward cache
          /*
          要用什么数据结构呢？
          可变数据结构还是不可变？
          可变？
           */
          backward(outputDelta.map(-_).memoize)
        }
          )
      })
    }


  }

}

object Dsl {

  trait Double {
    protected type Self >: this.type <: Double

    def unary_- : Self
  }

  object Double {
    type Aux[Self0] = Double {
      type Self = Self0
    }
  }

}

trait Dsl {

  import Dsl._

  type Any

  type Double <: Dsl.Double.Aux[Double] with Any
}

trait DeepLearning extends Dsl {
  self =>

  import DeepLearning._

  type InputData

  type Any = Differentiable.Aux[_, _, InputData, _, _]

  type Double = DifferentiableDouble[_, _, InputData]

}

//
//import shapeless.{HMap, ~?>}
//
//import scala.language.{existentials, higherKinds, implicitConversions}
//
//object DeepLearning {
//
//  type ForwardCache = HMap[ForwardCache.CacheRelation]
//
//  object ForwardCache {
//
//    final class CacheRelation[K, V] private[CacheRelation]()
//
//    object CacheRelation {
//      implicit def relation[A <: Any] = new CacheRelation[A, ForwardPass[A#Data, A#Delta]]
//    }
//
//  }
//
//  final case class ForwardPass[+Data, -Delta]
//  (outputData: Data, cache: ForwardCache)
//
//  trait Function1[-A0 <: Any, +R <: Any] extends Any {
//
//    type Data <: A0 => R
//
//    def forwardApply(input: A0, cache: ForwardCache) = {
//      // TODO: update cache
//      val dataForward = forward(cache)
//
//      dataForward.outputData(input).forward(dataForward.cache)
//    }
//  }
//
//  object Function1 {
//    override def apply[A0 <: Any, R <: Any](raw: (A0) => R): Function1[A0, R] = {
//      new Function1[A0, R] {
//        override def forward(cache: ForwardCache): ForwardPass[Data, Delta] = ???
//      }
//    }
//  }
//
//  trait Any {
//    type Data
//    type Delta
//
//    def forward(cache: ForwardCache): ForwardPass[Data, Delta]
//  }
//
//  object Double {
//    def apply(raw: scala.Double): Double = new Double {
//      override def forward(cache: ForwardCache) = ???
//    }
//  }
//
//  trait Double extends Any {
//    outer =>
//    type Data = cats.Eval[scala.Double]
//    type Delta = cats.Eval[scala.Double]
//
//    def +(other: Double): Double = ???
//
//    def /(other: Double): Double = ???
//
//    def *(other: Double): Double = ???
//
//    def -(other: Double): Double = ???
//
//    def unary_- : Double = new Double {
//
//      override def forward(cache: ForwardCache) = {
//        outer.forward(cache)
//        ???
//      }
//    }
//  }
//
//
//  override def exp(value: Double): Double = ???
//
//  override def log(value: Double): Double = ???
//
//  override def max(left: Double, right: Double): Double = ???
//
//
//  sealed trait HList extends Any {
//    override def ::[Head <: Any](head: Head): ::[Head, this.type] = ???
//  }
//
//  type HNil = HList
//
//  object HNil extends HList {
//    override def forward(cache: ForwardCache) = ???
//  }
//
//}
