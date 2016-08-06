package com.thoughtworks


import com.thoughtworks.Differentiable._
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

object DeepLearning {

  implicit object INDArrayPatch extends Patch[INDArray, Option[INDArray]] {
    override def applyPatch(weight: INDArray, patch: Option[INDArray], learningRate: Double): INDArray = {
      patch match {
        case None =>
          weight
        case Some(delta) =>
          weight + delta * learningRate
      }
    }

    override def combine(f1: Option[INDArray], f2: Option[INDArray]): Option[INDArray] = {
      f1 match {
        case None =>
          f2 match {
            case None => None
            case Some(f2Delta) => Some(f2Delta)

          }
        case Some(f1Delta) =>
          f2 match {
            case None => Some(f1Delta)
            case Some(f2Delta) => Some(f1Delta + f2Delta)
          }
      }
    }

    override def empty = None
  }

  /**
    * Workaround for https://github.com/milessabin/shapeless/issues/626
    */
  private[DeepLearning] val NeverChange = Differentiable.NeverChange

  trait PointfreeMultiply[=>:[_, _]] {
    def multiply: INDArray =>: INDArray =>: INDArray
  }

  final case class PartiallyAppliedMultiply
  (input0Data: INDArray, outer: Multiply.type)
  (implicit protected val inputPatch: Patch[INDArray, Option[INDArray]])
    extends DifferentiableFunction[INDArray, INDArray]
      with Cache[PartiallyAppliedMultiply, Option[INDArray], NeverChange.type]
      with PartiallyApplied[Option[INDArray], NeverChange.type] {

    type Self = PartiallyAppliedMultiply


    override implicit def patch: Patch[Self, Difference] = {
      Patch.genericPatch(
        Generic[Self],
        Generic[PartiallyApplied.PartiallyAppliedDifference[Option[INDArray], NeverChange.type]],
        Patch.hconsPatch(inputPatch, Patch.hconsPatch(outer.patch, Patch.HNilPatch))
      )
    }

    override def forward[InputData <: INDArray, InputDifference](input1: Differentiable.Aux[InputData, InputDifference]): Cache[INDArray, InputDifference, Difference] = {
      type ExpectedDifferentiable = Differentiable.Aux[_ <: INDArray, _ >: Option[INDArray]]
      input1 match {
        case differentiable1: ExpectedDifferentiable =>
          new Cache[INDArray, INDArray, Difference] {
            type OutputDifference = Option[INDArray]

            override def output = Differentiable(input0Data * differentiable1.self, INDArrayPatch)

            override def backward(difference: OutputDifference) = new Differences[INDArray, Difference] {
              override def inputDifference: INDArray = input0Data

              override def weightDifference: Difference = new Difference(Some(differentiable1.self), NeverChange)
            }
          }
      }
    }.unsafeCast
  }

  object Multiply extends DifferentiableFunction[INDArray, PartiallyAppliedMultiply] with PureFunction {

    override def forward[InputData <: INDArray, InputDifference](input0: Differentiable.Aux[InputData, InputDifference]): Cache[PartiallyAppliedMultiply, InputDifference, Difference] = {
      type ExpectedDifferentiable = Differentiable.Aux[_ <: INDArray, _ >: Option[INDArray]]
      input0 match {
        case differentiable0: ExpectedDifferentiable =>
          PartiallyAppliedMultiply(differentiable0.self, Multiply.this)
      }
    }.unsafeCast
  }

  implicit object DeepLearningInstances extends PointfreeMultiply[DifferentiableFunction] {
    override def multiply = Multiply
  }

}

