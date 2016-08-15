package com.thoughtworks


import cats.{Applicative, Eval, Monoid}
import com.thoughtworks.Differentiable.DifferentiableFunction.{AbstractDifferentiableFunction, BackwardPass, ForwardPass}
import com.thoughtworks.Differentiable.Pure.NoPatch
import com.thoughtworks.Differentiable.{DifferentiableFunction, _}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._


import scala.language.existentials
import scala.language.higherKinds

object DeepLearning {

  object DifferentiableINDArray {

    def evalInstances = Eval.now(INDArrayPatch)

    object INDArrayPatch extends Patch[INDArray, Option[INDArray]] with Monoid[Option[INDArray]] {
      override def apply(weight: INDArray, patch: Option[INDArray], learningRate: Double): INDArray = {
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

  }

  final case class DifferentiableINDArray(override val data: Eval[_ <: INDArray]) extends Differentiable[INDArray] {
    override type Data = INDArray
    override type Delta = Option[INDArray]
    override val monoid: Eval[_ <: Monoid[Delta]] = DifferentiableINDArray.evalInstances
    override val patch: Eval[_ <: Patch[Data, Delta]] = DifferentiableINDArray.evalInstances
  }

  trait PointfreeMultiply[F[_]] {
    def multiply: F[INDArray => INDArray => INDArray]
  }

  object Multiply extends DifferentiableFunction[INDArray, INDArray => INDArray] with Pure {
    override def forward[AData, ADelta] = {
      case DifferentiableINDArray(a) =>
        val partiallyAppled1 = new AbstractDifferentiableFunction(a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) with DifferentiableFunction[INDArray, INDArray] {
          override def forward[InputData, InputDelta] = {
            case DifferentiableINDArray(b) =>
              val output = Applicative[Eval].map2(a, b){ (aData:INDArray, bData:INDArray) =>
                aData * bData
              }
              ForwardPass(DifferentiableINDArray(output), {outputDifference: Eval[_ <: Option[INDArray]] =>
                BackwardPass(b.map(Some(_)), a.map(Some(_)))
              })
          }
        }
        ForwardPass(partiallyAppled1, { outputDifference: Eval[_ <: Option[INDArray]] =>
          BackwardPass(NoPatch.eval, outputDifference)
        })
    }
  }

  implicit object DeepLearningInstances extends PointfreeMultiply[Differentiable] {
    override def multiply = Multiply
  }

}

