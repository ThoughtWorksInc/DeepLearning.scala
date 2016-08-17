package com.thoughtworks


import cats.data.{Cokleisli, Kleisli}
import cats.kernel.std.DoubleGroup
import cats.{Applicative, Eval, Monoid}
import com.thoughtworks.Differentiable.DifferentiableFunction.{AbstractDifferentiableFunction, BackwardPass, ForwardPass}
import com.thoughtworks.Differentiable.Pure.NoPatch
import com.thoughtworks.Differentiable.{DifferentiableFunction, _}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import simulacrum.typeclass

import scala.language.{existentials, higherKinds, implicitConversions}

object DeepLearning {

  object DifferentiableDouble {

    object DoublePatch extends DoubleGroup with Patch[Double, Double] {
      override def apply(data: Double, delta: Double, learningRate: Double): Double = {
        data + delta * learningRate
      }
    }

    def evalInstances = Eval.now(DoublePatch)

    implicit val lift = Kleisli { value: Double =>
      DifferentiableDouble(Eval.now(value))
    }

    implicit val unlift = Cokleisli[Differentiable, Double, Double] {
      case DifferentiableDouble(eval) => eval.value
    }

  }

  final case class DifferentiableDouble(override val data: Eval[_ <: Double]) extends Differentiable[Double] {
    override type Data = Double
    override type Delta = Double
    override val monoid: Eval[_ <: Monoid[Delta]] = DifferentiableDouble.evalInstances
    override val patch: Eval[_ <: Patch[Data, Delta]] = DifferentiableDouble.evalInstances
  }

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

  @typeclass
  trait PointfreeMaximum[F[_]] {
    def max: F[INDArray => Double => INDArray]
  }

  object Max extends DifferentiableFunction[INDArray, Double => INDArray] with Pure {
    override def forward[InputData, InputDelta] = {
      case DifferentiableINDArray(a) =>
        val partiallyAppled1 = new AbstractDifferentiableFunction(a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) with DifferentiableFunction[Double, INDArray] {
          override def forward[BData, BDelta] = {
            case DifferentiableDouble(b) =>
              val output = Applicative[Eval].map2(a, b) { (aData: INDArray, bData: Double) =>
                Transforms.max(aData, bData)
              }.memoize
              ForwardPass(DifferentiableINDArray(output), { outputDifference: Eval[_ <: Option[INDArray]] =>
                BackwardPass(
                  outputDifference.flatMap[Option[INDArray]] {
                    case None => Eval.now(None)
                    case Some(outputDelta) =>
                      Applicative[Eval].map2(a, b) { (aData: INDArray, bData: Double) =>
                        Some((aData gt bData) * outputDelta)
                      }
                  },
                  outputDifference.flatMap[Double] {
                    case None => Eval.now(0)
                    case Some(outputDelta) =>
                      Applicative[Eval].map2(a, b) { (aData: INDArray, bData: Double) =>
                        ((aData lt bData) * outputDelta).sumT
                      }
                  }
                )
              })
          }
        }
        ForwardPass(partiallyAppled1, { outputDifference: Eval[_ <: Option[INDArray]] =>
          BackwardPass(NoPatch.eval, outputDifference)
        })
    }
  }


  @typeclass
  trait PointfreeNegative[F[_]] {
    def neg: F[INDArray => INDArray]
  }

  object Neg extends DifferentiableFunction[INDArray, INDArray] with Pure {
    override def forward[InputData, InputDelta] = {
      case DifferentiableINDArray(a) =>
        val evalOutput = a.map(_.neg)
        ForwardPass(DifferentiableINDArray(evalOutput), { outputDifference: Eval[_ <: Option[INDArray]] =>
          BackwardPass(NoPatch.eval, outputDifference.map[Option[INDArray]](_.map(_.neg)))
        })
    }
  }

  @typeclass
  trait PointfreeExponentiation[F[_]] {
    def exp: F[INDArray => INDArray]
  }

  object Exp extends DifferentiableFunction[INDArray, INDArray] with Pure {
    override def forward[InputData, InputDelta] = {
      case DifferentiableINDArray(a) =>
        val evalOutput = a.map(Transforms.exp)
        ForwardPass(DifferentiableINDArray(evalOutput), { outputDifference: Eval[_ <: Option[INDArray]] =>
          BackwardPass(NoPatch.eval, outputDifference.flatMap[Option[INDArray]] {
            case None =>
              Eval.now(None)
            case Some(outputDelta) => evalOutput.map { output: INDArray =>
              Some(output * outputDelta)
            }
          })
        })
    }
  }

  @typeclass
  trait PointfreeAddition[F[_]] {
    def add: F[INDArray => INDArray => INDArray]
  }

  object Add extends DifferentiableFunction[INDArray, INDArray => INDArray] with Pure {
    override def forward[AData, ADelta] = {
      case DifferentiableINDArray(a) =>
        val partiallyAppled1 = new AbstractDifferentiableFunction(a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) with DifferentiableFunction[INDArray, INDArray] {
          override def forward[BData, BDelta] = {
            case DifferentiableINDArray(b) =>
              val output = Applicative[Eval].map2(a, b) { (aData: INDArray, bData: INDArray) =>
                aData + bData
              }.memoize
              ForwardPass(DifferentiableINDArray(output), { outputDifference: Eval[_ <: Option[INDArray]] =>
                BackwardPass(outputDifference, outputDifference)
              })
          }
        }
        ForwardPass(partiallyAppled1, { outputDifference: Eval[_ <: Option[INDArray]] =>
          BackwardPass(NoPatch.eval, outputDifference)
        })
    }
  }

  @typeclass
  trait PointfreeMultiplication[F[_]] {
    def mul: F[INDArray => INDArray => INDArray]
  }

  object Mul extends DifferentiableFunction[INDArray, INDArray => INDArray] with Pure {
    override def forward[AData, ADelta] = {
      case DifferentiableINDArray(a) =>
        val partiallyAppled1 = new AbstractDifferentiableFunction(a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) with DifferentiableFunction[INDArray, INDArray] {
          override def forward[BData, BDelta] = {
            case DifferentiableINDArray(b) =>
              val output = Applicative[Eval].map2(a, b) { (aData: INDArray, bData: INDArray) =>
                aData * bData
              }.memoize
              ForwardPass(DifferentiableINDArray(output), { outputDifference: Eval[_ <: Option[INDArray]] =>
                BackwardPass(
                  outputDifference.flatMap[Option[INDArray]] {
                    case None => Eval.now(None)
                    case Some(outputDelta) =>
                      b.map { bData: INDArray =>
                        Some(bData * outputDelta)
                      }
                  },
                  outputDifference.flatMap[Option[INDArray]] {
                    case None => Eval.now(None)
                    case Some(outputDelta) =>
                      a.map { aData: INDArray =>
                        Some(aData * outputDelta)
                      }
                  }
                )
              })
          }
        }
        ForwardPass(partiallyAppled1, { outputDifference: Eval[_ <: Option[INDArray]] =>
          BackwardPass(NoPatch.eval, outputDifference)
        })
    }
  }

  @typeclass
  trait PointfreeDot[F[_]] {
    def dot: F[INDArray => INDArray => INDArray]
  }

  object Dot extends DifferentiableFunction[INDArray, INDArray => INDArray] with Pure {
    override def forward[AData, ADelta] = {
      case DifferentiableINDArray(a) =>
        val partiallyAppled1 = new AbstractDifferentiableFunction(a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) with DifferentiableFunction[INDArray, INDArray] {
          override def forward[BData, BDelta] = {
            case DifferentiableINDArray(b) =>
              val output = Applicative[Eval].map2(a, b) { (aData: INDArray, bData: INDArray) =>
                aData.dot(bData)
              }.memoize
              ForwardPass(DifferentiableINDArray(output), { outputDifference: Eval[_ <: Option[INDArray]] =>
                BackwardPass(
                  outputDifference.flatMap[Option[INDArray]] {
                    case None => Eval.now(None)
                    case Some(outputDelta) =>
                      a.map { aData =>
                        Some(aData.T.dot(outputDelta))
                      }
                  },
                  outputDifference.flatMap[Option[INDArray]] {
                    case None => Eval.now(None)
                    case Some(outputDelta) =>
                      b.map { bData =>
                        Some(outputDelta.dot(bData.T))
                      }
                  }
                )
              })
          }
        }
        ForwardPass(partiallyAppled1, { outputDifference: Eval[_ <: Option[INDArray]] =>

          BackwardPass(NoPatch.eval, outputDifference)
        })
    }
  }

  @typeclass
  trait PointfreeDeepLearning[F[_]] extends PointfreeExponentiation[F] with PointfreeNegative[F] with PointfreeAddition[F] with PointfreeMultiplication[F] with PointfreeDot[F] with PointfreeMaximum[F] {

    final def relu(implicit pointfree: Pointfree[F], freezing: Freezing[F], liftDouble: Kleisli[F, Double, Double]) = {
      import pointfree._
      import freezing._
      import Pointfree.ops._
      flip[INDArray, Double, INDArray] ap max ap (freeze[Double] ap liftDouble(0.0))
    }

    final def fullyConnected(weight: F[INDArray], bias: F[INDArray])(implicit pointfree: Pointfree[F]) = {
      import pointfree._
      import Pointfree.ops._
      pointfree.arrow.andThen[INDArray, INDArray, INDArray](flip[INDArray, INDArray, INDArray] ap dot ap weight, add ap bias)
    }

    final def fullyConnectedThenRelu(inputSize: Int, outputSize: Int)(implicit pointfree: Pointfree[F], freezing: Freezing[F], liftDouble: Kleisli[F, Double, Double], liftINDArray: Kleisli[F, INDArray, INDArray]) = {
      import pointfree._
      import Pointfree.ops._
      val fc = fullyConnected(liftINDArray(Nd4j.randn(inputSize, outputSize) / math.sqrt(inputSize / 2)), liftINDArray(Nd4j.zeros(outputSize)))
      arrow.andThen[INDArray, INDArray, INDArray](fc, relu)
    }

  }

  implicit object DeepLearningInstances extends PointfreeDeepLearning[Differentiable] {
    override def mul = Mul

    override def dot = Dot

    override def add = Add

    override def max = Max

    override def exp = Exp

    override def neg = Neg

  }

}

