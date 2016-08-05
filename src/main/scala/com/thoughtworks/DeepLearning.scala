package com.thoughtworks


import com.thoughtworks.Differentiable._
import com.thoughtworks.Differentiable.Patch.PairPatch
import com.thoughtworks.Differentiable.DifferentiableFunction
import com.thoughtworks.Differentiable.DifferentiableFunction._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms._
import shapeless.{::, DepFn0, DepFn1, DepFn2, Generic, HList, HNil, Poly0, PolyApply, Widen, the}

import scala.language.existentials
import scala.language.higherKinds
import scalaz.syntax.Ops
import scalaz.{-\/, Apply, Arrow, Category, Choice, Compose, Lens, Monoid, Semigroup, Split, Strong, \/, \/-}

object DeepLearning {

  /**
    * Workaround for https://github.com/milessabin/shapeless/issues/626
    */
  private[DeepLearning] val NeverChange = Differentiable.NeverChange

  trait PointfreeMultiply[=>:[_, _]] {
    def multiply: INDArray =>: INDArray =>: INDArray
  }

  object PartialApplied {

    final case class PartialAppliedDifference[InputDifference, FDifference]
    (inputDifference: InputDifference, weightDifference: FDifference)
      extends Differences[InputDifference, FDifference]

  }


  trait PartialApplied[InputDifference0, FDifference] {
    _: DifferentiableFunction[_, _] with Cache[_, InputDifference0, FDifference] =>

    type Difference = PartialApplied.PartialAppliedDifference[InputDifference0, FDifference]

    override def output: Self = this

    type OutputDifference = Difference

    override def backward(difference: Difference): Difference = difference

  }

  final case class PartialAppliedMultiply
  (input0Data: INDArray, outer: Multiply.type)
  (implicit protected val inputPatch: Patch[INDArray, Option[INDArray]])
    extends DifferentiableFunction[INDArray, INDArray]
      with Cache[PartialAppliedMultiply, Option[INDArray], NeverChange.type]
      with PartialApplied[Option[INDArray], NeverChange.type] {

    type Self = PartialAppliedMultiply


    override implicit def patch: Patch[Self, Difference] = {
      Patch.genericPatch(
        Generic[Self],
        Generic[DeepLearning.PartialApplied.PartialAppliedDifference[Option[INDArray], NeverChange.type]],
        Patch.hconsPatch(inputPatch, Patch.hconsPatch(outer.patch, Patch.HNilPatch))
      )
    }

    override def forward[InputData <: INDArray, InputDifference](input1: Differentiable.Aux[InputData, InputDifference]): Cache[INDArray, InputDifference, Difference] = {
      type ExpectedDifferentiable = Differentiable.Aux[_ <: INDArray, _ >: Option[INDArray]]
      input1 match {
        case differentiable1: ExpectedDifferentiable =>
          new Cache[INDArray, INDArray, Difference] {
            type OutputDifference = Option[INDArray]

            override def output = PatchOps(input0Data * differentiable1.self, Patch.INDArrayPatch)

            override def backward(difference: OutputDifference) = new Differences[INDArray, Difference] {
              override def inputDifference: INDArray = input0Data

              override def weightDifference: Difference = new Difference(Some(differentiable1.self), NeverChange)
            }
          }
      }
    }.unsafeCast
  }

  object Multiply extends DifferentiableFunction[INDArray, PartialAppliedMultiply] with PureFunction {

    override def forward[InputData <: INDArray, InputDifference](input0: Differentiable.Aux[InputData, InputDifference]): Cache[PartialAppliedMultiply, InputDifference, Difference] = {
      type ExpectedDifferentiable = Differentiable.Aux[_ <: INDArray, _ >: Option[INDArray]]
      input0 match {
        case differentiable0: ExpectedDifferentiable =>
          PartialAppliedMultiply(differentiable0.self, Multiply.this)
      }
    }.unsafeCast
  }

  implicit object DeepLearningInstances extends PointfreeMultiply[DifferentiableFunction] {
    override def multiply = Multiply
  }

}

