package com.thoughtworks


import cats.data.{Cokleisli, Kleisli}
import cats.kernel.std.DoubleGroup
import cats.{Applicative, Eval, EvalMonoid, Monoid}
import com.thoughtworks.DeepLearning.BinaryOperator.OperatorDispatcher
import com.thoughtworks.Differentiable.DifferentiableFunction.{BackwardPass, ForwardPass}
import com.thoughtworks.Differentiable._
import com.thoughtworks.Pointfree.Ski
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import simulacrum.{op, typeclass}

import scala.language.{existentials, higherKinds, implicitConversions}

// Disable + for concatenation
import Predef.{any2stringadd => _, _}

object DeepLearning {

  type Array2D = Array[Array[Double]]

  implicit def implicitLiftArray2D[F[_] : PointfreeOperators](value: Array2D): F[Array2D] = {
    PointfreeOperators[F].liftArray2D(value)
  }

  implicit def implicitLiftDouble[F[_] : PointfreeOperators](value: Double): F[Double] = {
    PointfreeOperators[F].liftDouble(value)
  }

  case object DifferentiableDouble extends Differentiable[Eval[Double]] {

    object DifferentiableDoubleInstances extends EvalMonoid[Double] with Patch[Eval[Double], Eval[Double]] {


      override implicit def algebra = cats.std.double.doubleGroup

      override def apply(data: Eval[Double], delta: Eval[Double], learningRate: Double): Eval[Double] = {
        Applicative[Eval].map2(data, delta)(_ + _ * learningRate).memoize
      }
    }

    override type Delta = Eval[Double]

    override def monoid = DifferentiableDoubleInstances

    override def patch = DifferentiableDoubleInstances
  }

  case object DifferentiableINDArray extends Differentiable[Eval[INDArray]] {

    override type Delta = Eval[Option[INDArray]]


    override def monoid = DifferentiableINDArrayInstances

    override def patch = DifferentiableINDArrayInstances

    object DifferentiableINDArrayInstances extends Patch[Eval[INDArray], Eval[Option[INDArray]]] with EvalMonoid[Option[INDArray]] {

      override implicit object algebra extends Monoid[Option[INDArray]] {
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

      override def apply(weight: Eval[INDArray], delta: Eval[Option[INDArray]], learningRate: Double): Eval[INDArray] = {
        delta.flatMap {
          case None => weight
          case Some(deltaValue) => weight.map(_ + deltaValue * learningRate)
        }.memoize
      }
    }

  }


  //
  //  def Max = DifferentiableFunction pure new PureForward[Array2D, Double => Array2D] {
  //    override def apply[InputData, InputDelta] = {
  //      case (_, DifferentiableINDArray(a)) =>
  //        val partiallyAppled1 = DifferentiableFunction.Factory[Double, Array2D](a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) { factory => new factory.Forward {
  //          override def apply[BData, BDelta] = {
  //            case (DifferentiableFunction(a, _, _, _), DifferentiableDouble(b)) =>
  //              val output = Applicative[Eval].map2(a, b) { (aData: INDArray, bData: Double) =>
  //                Transforms.max(aData, bData)
  //              }.memoize
  //              ForwardPass(DifferentiableINDArray(output), { outputDifference: Eval[_ <: Option[INDArray]] =>
  //                BackwardPass(
  //                  outputDifference.flatMap[Option[INDArray]] {
  //                    case None => Eval.now(None)
  //                    case Some(outputDelta) =>
  //                      Applicative[Eval].map2(a, b) { (aData: INDArray, bData: Double) =>
  //                        Some((aData gt bData) * outputDelta)
  //                      }
  //                  },
  //                  outputDifference.flatMap[Double] {
  //                    case None => Eval.now(0)
  //                    case Some(outputDelta) =>
  //                      Applicative[Eval].map2(a, b) { (aData: INDArray, bData: Double) =>
  //                        ((aData lt bData) * outputDelta).sumT
  //                      }
  //                  }
  //                )
  //              })
  //          }
  //        }
  //        }
  //        ForwardPass(partiallyAppled1, { outputDifference: Eval[_ <: Option[INDArray]] =>
  //          BackwardPass(NoPatch.eval, outputDifference)
  //        })
  //    }
  //  }
  //
  //
  //  def Negative = DifferentiableFunction pure new PureForward[Array2D, Array2D] {
  //    override def apply[InputData, InputDelta] = {
  //      case (_, DifferentiableINDArray(a)) =>
  //        val evalOutput = a.map(_.neg)
  //        ForwardPass(DifferentiableINDArray(evalOutput), { outputDifference: Eval[_ <: Option[INDArray]] =>
  //          BackwardPass(NoPatch.eval, outputDifference.map[Option[INDArray]](_.map(_.neg)))
  //        })
  //    }
  //  }
  //
  //  def ReciprocalDouble = DifferentiableFunction pure new PureForward[Double, Double] {
  //    override def apply[InputData, InputDelta] = {
  //      case (_, DifferentiableDouble(a)) =>
  //        val evalOutput = a.map(1 / _)
  //        ForwardPass(DifferentiableDouble(evalOutput), { outputDifference: Eval[_ <: Double] =>
  //          BackwardPass(NoPatch.eval, Applicative[Eval].map2(outputDifference, outputDifference) { (outputDelta: Double, aValue: Double) =>
  //            -outputDelta / (aValue * aValue)
  //          })
  //        })
  //    }
  //  }
  //
  //  def ReciprocalArray2D = DifferentiableFunction pure new PureForward[Array2D, Array2D] {
  //    override def apply[InputData, InputDelta] = {
  //      case (_, DifferentiableINDArray(a)) =>
  //        val evalOutput = a.map(_ rdiv 1)
  //        ForwardPass(DifferentiableINDArray(evalOutput), { outputDifference: Eval[_ <: Option[INDArray]] =>
  //
  //          BackwardPass(NoPatch.eval, outputDifference.flatMap[Option[INDArray]] {
  //            case None =>
  //              Eval.now(None)
  //            case Some(outputDelta) =>
  //              a.map { aValue: INDArray =>
  //                Some(-outputDelta / (aValue * aValue))
  //              }
  //          })
  //        })
  //    }
  //  }
  //
  //
  //  def Exp = DifferentiableFunction pure new PureForward[Array2D, Array2D] {
  //    override def apply[InputData, InputDelta] = {
  //      case (_, DifferentiableINDArray(a)) =>
  //        val evalOutput = a.map(Transforms.exp)
  //        ForwardPass(DifferentiableINDArray(evalOutput), { outputDifference: Eval[_ <: Option[INDArray]] =>
  //          BackwardPass(NoPatch.eval, outputDifference.flatMap[Option[INDArray]] {
  //            case None =>
  //              Eval.now(None)
  //            case Some(outputDelta) => evalOutput.map { output: INDArray =>
  //              Some(output * outputDelta)
  //            }
  //          })
  //        })
  //    }
  //  }
  //
  //  def AddArrayDouble = DifferentiableFunction pure new PureForward[Array2D, Double => Array2D] {
  //    override def apply[AData, ADelta] = {
  //      case (_, DifferentiableINDArray(a)) =>
  //        val partiallyAppled1 = DifferentiableFunction.Factory[Double, Array2D](a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) { factory => new factory.Forward {
  //          override def apply[BData, BDelta] = {
  //            case (DifferentiableFunction(a, _, _, _), DifferentiableDouble(b)) =>
  //              val output = Applicative[Eval].map2(a, b) { (aData: INDArray, bData: Double) =>
  //                aData + bData
  //              }.memoize
  //              ForwardPass(DifferentiableINDArray(output), { outputDifference: Eval[_ <: Option[INDArray]] =>
  //                val aDelta: Eval[_ <: Option[INDArray]] = outputDifference
  //                val bDelta: Eval[_ <: Double] = outputDifference.map {
  //                  case None => 0
  //                  case Some(od) => od.sumT
  //                }.memoize
  //                BackwardPass(aDelta, bDelta)
  //              })
  //          }
  //        }
  //        }
  //        ForwardPass(partiallyAppled1, { outputDifference: Eval[_ <: Option[INDArray]] =>
  //          BackwardPass(NoPatch.eval, outputDifference)
  //        })
  //    }
  //  }
  //
  //  def MultiplyArray2DDouble = DifferentiableFunction pure new PureForward[Array2D, Double => Array2D] {
  //    override def apply[AData, ADelta] = {
  //      case (_, DifferentiableINDArray(a)) =>
  //        val partiallyAppled1 = DifferentiableFunction.Factory[Double, Array2D](a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) { factory => new factory.Forward {
  //          override def apply[BData, BDelta] = {
  //            case (DifferentiableFunction(a, _, _, _), DifferentiableDouble(b)) =>
  //              val output = Applicative[Eval].map2(a, b) { (aData: INDArray, bData: Double) =>
  //                aData * bData
  //              }.memoize
  //              ForwardPass(DifferentiableINDArray(output), { outputDifference: Eval[_ <: Option[INDArray]] =>
  //                val aDelta: Eval[_ <: Option[INDArray]] = outputDifference.flatMap[Option[INDArray]] {
  //                  case None => Eval.now(None)
  //                  case Some(outputDelta) =>
  //                    b.map { bData: Double =>
  //                      Some(outputDelta * bData)
  //                    }
  //                }.memoize
  //                val bDelta: Eval[_ <: Double] = outputDifference.flatMap[Double] {
  //                  case None => Eval.now(0.0)
  //                  case Some(outputDelta) =>
  //                    a.map { aData: INDArray =>
  //                      (aData * outputDelta).sumT
  //                    }
  //                }.memoize
  //                BackwardPass(aDelta, bDelta)
  //              })
  //          }
  //        }
  //        }
  //        ForwardPass(partiallyAppled1, { outputDifference: Eval[_ <: Option[INDArray]] =>
  //          BackwardPass(NoPatch.eval, outputDifference)
  //        })
  //    }
  //  }
  //
  //  def AddArrayArray = DifferentiableFunction pure new PureForward[Array2D, Array2D => Array2D] {
  //    override def apply[AData, ADelta] = {
  //      case (_, DifferentiableINDArray(a)) =>
  //        val partiallyAppled1 = DifferentiableFunction.Factory[Array2D, Array2D](a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) { factory => new factory.Forward {
  //          override def apply[BData, BDelta] = {
  //            case (DifferentiableFunction(a, _, _, _), DifferentiableINDArray(b)) =>
  //              val output = Applicative[Eval].map2(a, b) { (aData: INDArray, bData: INDArray) =>
  //                aData + bData
  //              }.memoize
  //              ForwardPass(DifferentiableINDArray(output), { outputDifference: Eval[_ <: Option[INDArray]] =>
  //                BackwardPass(outputDifference, outputDifference)
  //              })
  //          }
  //        }
  //        }
  //        ForwardPass(partiallyAppled1, { outputDifference: Eval[_ <: Option[INDArray]] =>
  //          BackwardPass(NoPatch.eval, outputDifference)
  //        })
  //    }
  //  }
  //
  //  def MultiplyArray2DArray2D = DifferentiableFunction pure new PureForward[Array2D, Array2D => Array2D] {
  //    override def apply[AData, ADelta] = {
  //      case (_, DifferentiableINDArray(a)) =>
  //        val partiallyAppled1 = DifferentiableFunction.Factory[Array2D, Array2D](a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) { factory => new factory.Forward {
  //          override def apply[BData, BDelta] = {
  //            case (DifferentiableFunction(a, _, _, _), DifferentiableINDArray(b)) =>
  //              val output = Applicative[Eval].map2(a, b) { (aData: INDArray, bData: INDArray) =>
  //                aData * bData
  //              }.memoize
  //              ForwardPass(DifferentiableINDArray(output), { outputDifference: Eval[_ <: Option[INDArray]] =>
  //                BackwardPass(
  //                  outputDifference.flatMap[Option[INDArray]] {
  //                    case None => Eval.now(None)
  //                    case Some(outputDelta) =>
  //                      b.map { bData: INDArray =>
  //                        Some(bData * outputDelta)
  //                      }
  //                  },
  //                  outputDifference.flatMap[Option[INDArray]] {
  //                    case None => Eval.now(None)
  //                    case Some(outputDelta) =>
  //                      a.map { aData: INDArray =>
  //                        Some(aData * outputDelta)
  //                      }
  //                  }
  //                )
  //              })
  //          }
  //        }
  //        }
  //        ForwardPass(partiallyAppled1, { outputDifference: Eval[_ <: Option[INDArray]] =>
  //          BackwardPass(NoPatch.eval, outputDifference)
  //        })
  //    }
  //  }
  //
  //  def Dot = DifferentiableFunction pure new PureForward[Array2D, Array2D => Array2D] {
  //    override def apply[AData, ADelta] = {
  //      case (_, DifferentiableINDArray(a)) =>
  //        val partiallyAppled1 = DifferentiableFunction.Factory[Array2D, Array2D](a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) { factory => new factory.Forward {
  //          override def apply[BData, BDelta] = {
  //            case (DifferentiableFunction(a, _, _, _), DifferentiableINDArray(b)) =>
  //              val output = Applicative[Eval].map2(a, b) { (aData: INDArray, bData: INDArray) =>
  //                aData.dot(bData)
  //              }.memoize
  //              ForwardPass(DifferentiableINDArray(output), { outputDifference: Eval[_ <: Option[INDArray]] =>
  //                BackwardPass(
  //                  outputDifference.flatMap[Option[INDArray]] {
  //                    case None => Eval.now(None)
  //                    case Some(outputDelta) =>
  //                      a.map { aData =>
  //                        Some(aData.T.dot(outputDelta))
  //                      }
  //                  },
  //                  outputDifference.flatMap[Option[INDArray]] {
  //                    case None => Eval.now(None)
  //                    case Some(outputDelta) =>
  //                      b.map { bData =>
  //                        Some(outputDelta.dot(bData.T))
  //                      }
  //                  }
  //                )
  //              })
  //          }
  //        }
  //        }
  //        ForwardPass(partiallyAppled1, { outputDifference: Eval[_ <: Option[INDArray]] =>
  //          BackwardPass(NoPatch.eval, outputDifference)
  //        })
  //    }
  //  }

  implicit def kleisliWithParameter[F[_], A, B, Parameter](implicit ski: Ski[F], lift: Kleisli[F, A, B]): Kleisli[Lambda[X => F[Parameter => X]], A, B] = Kleisli[Lambda[X => F[Parameter => X]], A, B] { a: A =>
    import Ski.ops._
    lift(a).withParameter[Parameter]
  }

  @typeclass
  trait BinaryOperator[F[_]] {
    def doubleArray(left: F[Double], right: F[Array2D]): F[Array2D]

    def arrayArray(left: F[Array2D], right: F[Array2D]): F[Array2D]

    def doubleDouble(left: F[Double], right: F[Double]): F[Double]

    def arrayDouble(left: F[Array2D], right: F[Double]): F[Array2D]
  }

  object BinaryOperator {

    trait OperatorDispatcher[Left, Right] {
      type Out

      def apply[F[_] : BinaryOperator](left: F[Left], right: F[Right]): F[Out]
    }

    implicit object Array2DArray2DAddOperator extends OperatorDispatcher[Array2D, Array2D] {
      override type Out = Array2D

      override def apply[F[_] : BinaryOperator](left: F[Array2D], right: F[Array2D]): F[Array2D] = {
        BinaryOperator[F].arrayArray(left, right)
      }
    }

    implicit object DoubleDoubleAddOperator extends OperatorDispatcher[Double, Double] {
      override type Out = Double

      override def apply[F[_] : BinaryOperator](left: F[Double], right: F[Double]): F[Double] = {
        BinaryOperator[F].doubleDouble(left, right)
      }
    }

    implicit object DoubleArray2DAddOperator extends OperatorDispatcher[Double, Array2D] {
      override type Out = Array2D

      override def apply[F[_] : BinaryOperator](left: F[Double], right: F[Array2D]): F[Array2D] = {
        BinaryOperator[F].doubleArray(left, right)
      }
    }

    implicit object Array2DDoubleAddOperator extends OperatorDispatcher[Array2D, Double] {
      override type Out = Array2D

      override def apply[F[_] : BinaryOperator](left: F[Array2D], right: F[Double]): F[Array2D] = {
        BinaryOperator[F].arrayDouble(left, right)
      }
    }

  }


  @typeclass
  trait PointfreeOperators[F[_]] extends PointfreeFreezing[F] {

    def liftDouble: Kleisli[F, Double, Double]

    def liftArray2D: Kleisli[F, Array2D, Array2D]

    def zeros(numberOfRows: Int, numberOfColumns: Int): F[Array2D]

    def randn(numberOfRows: Int, numberOfColumns: Int): F[Array2D]

    def reciprocalDouble: F[Double => Double]

    def reciprocalArray2D: F[Array2D => Array2D]

    def negative: F[Array2D => Array2D]

    def multiplyArray2DArray2D: F[(Array2D, Array2D) => Array2D]

    def multiplyArray2DDouble: F[(Array2D, Double) => Array2D]

    def multiplyDoubleDouble: F[(Double, Double) => Double]

    def addArray2DArray2D: F[(Array2D, Array2D) => Array2D]

    def addArray2DDouble: F[(Array2D, Double) => Array2D]

    def addDoubleDouble: F[(Double, Double) => Double]

    def dot: F[(Array2D, Array2D) => Array2D]

    def max: F[(Array2D, Double) => Array2D]

    def exp: F[Array2D => Array2D]

    def sqrtDouble: F[Double => Double]

    def sqrtArray2D: F[Array2D => Array2D]

    implicit private def self = this

    import PointfreeOperators.ops._

    def +[A, B](left: F[A], right: F[B])(implicit dispatcher: OperatorDispatcher[A, B]): F[dispatcher.Out] = {
      dispatcher(left, right)(new BinaryOperator[F] {
        override def doubleArray(left: F[Double], right: F[Array2D]): F[Array2D] = {
          addArray2DDouble(right, left)
        }

        override def arrayArray(left: F[Array2D], right: F[Array2D]): F[Array2D] = {
          addArray2DArray2D(left, right)
        }

        override def doubleDouble(left: F[Double], right: F[Double]): F[Double] = {
          addDoubleDouble(left, right)
        }

        override def arrayDouble(left: F[Array2D], right: F[Double]): F[Array2D] = {
          addArray2DDouble(left, right)
        }
      })
    }

    def -[A](left: F[A], right: F[Array2D])(implicit constrait: F[A] <:< F[Array2D]) = {
      addArray2DArray2D(constrait(left), (negative(right)))
    }

    def *[A](left: F[A], right: F[Array2D])(implicit constrait: F[A] <:< F[Array2D]) = {
      multiplyArray2DArray2D(constrait(left), right)
    }

    def /[A, B](left: F[A], right: F[B])(implicit dispatcher: OperatorDispatcher[A, B]): F[dispatcher.Out] = {
      dispatcher(left, right)(new BinaryOperator[F] {
        override def doubleArray(left: F[Double], right: F[Array2D]): F[Array2D] = {
          multiplyArray2DDouble((reciprocalArray2D(right)), left)
        }

        override def arrayArray(left: F[Array2D], right: F[Array2D]): F[Array2D] = {
          multiplyArray2DArray2D(left, (reciprocalArray2D(right)))
        }

        override def doubleDouble(left: F[Double], right: F[Double]): F[Double] = {
          multiplyDoubleDouble(left, (reciprocalDouble(right)))
        }

        override def arrayDouble(left: F[Array2D], right: F[Double]): F[Array2D] = {
          multiplyArray2DDouble(left, (reciprocalDouble(right)))
        }
      })
    }

    def unary_-[A](value: F[A])(implicit constrait: F[A] <:< F[Array2D]) = {
      negative(constrait(value))
    }

  }

  object PointfreeOperators {

    import PointfreeOperators.ops._

    trait WithParameter[F[_], Parameter] extends PointfreeFreezing.WithParameter[F, Parameter] with PointfreeOperators[Lambda[X => F[Parameter => X]]] {
      implicit protected def outer: PointfreeOperators[F]

      override def liftDouble = kleisliWithParameter(outer, outer.liftDouble)

      override def liftArray2D = kleisliWithParameter(outer, outer.liftArray2D)

      override def sqrtDouble = outer.sqrtDouble.withParameter

      override def sqrtArray2D = outer.sqrtArray2D.withParameter

      override def multiplyArray2DArray2D = outer.multiplyArray2DArray2D.withParameter

      override def multiplyArray2DDouble = outer.multiplyArray2DDouble.withParameter

      override def multiplyDoubleDouble = outer.multiplyDoubleDouble.withParameter

      override def addArray2DArray2D = outer.addArray2DArray2D.withParameter

      override def addArray2DDouble = outer.addArray2DDouble.withParameter

      override def addDoubleDouble = outer.addDoubleDouble.withParameter

      override def negative = outer.negative.withParameter

      override def reciprocalArray2D = outer.reciprocalArray2D.withParameter

      override def reciprocalDouble = outer.reciprocalDouble.withParameter

      override def max = (outer: PointfreeOperators[F]).max.withParameter

      override def exp = (outer: PointfreeOperators[F]).exp.withParameter

      override def dot = (outer: PointfreeOperators[F]).dot.withParameter

      override def zeros(numberOfRows: Int, numberOfColumns: Int): F[(Parameter) => Array2D] = outer.zeros(numberOfRows, numberOfColumns).withParameter

      override def randn(numberOfRows: Int, numberOfColumns: Int): F[(Parameter) => Array2D] = outer.randn(numberOfRows, numberOfColumns).withParameter
    }

    implicit def withParameterInstances[F[_], Parameter](implicit underlying: PointfreeOperators[F]) = new WithParameter[F, Parameter] {
      override implicit protected def outer = underlying
    }

  }

  implicit object DeepLearningInstances extends PointfreeOperators[WeakOps] with DifferentiableInstances {
    //    override def multiplyArray2DArray2D = MultiplyArray2DArray2D
    //
    //    override def multiplyArray2DDouble = MultiplyArray2DDouble
    //
    //    override def multiplyDoubleDouble: Differentiable[(Double) => (Double) => Double] = ???
    //
    //    override def dot = Dot
    //
    //    override def addArray2DArray2D = AddArrayArray
    //
    //    override def addArray2DDouble = AddArrayDouble //: Differentiable[(INDArray) => (Double) => INDArray] = ???
    //
    //    override def addDoubleDouble: Differentiable[(Double) => (Double) => Double] = ???
    //
    //    override def max = Max
    //
    //    override def exp = Exp
    //
    //    override def negative = Negative
    //
    //    override def reciprocalArray2D = ReciprocalArray2D
    //
    //    override def reciprocalDouble = ReciprocalDouble
    //
    //    override def liftDouble: Kleisli[Differentiable, Double, Double] = DifferentiableDouble.lift
    //
    //    override def liftArray2D = DifferentiableINDArray.liftArray2D
    //
    //    override def liftINDArray = DifferentiableINDArray.lift

    override val liftDouble = Kleisli[WeakOps, Double, Double] { value: Double =>
      StrongOps[Eval[Double], Eval[Double], DifferentiableDouble.type](Eval.now(value), DifferentiableDouble).erase[Double]
    }

    override val liftArray2D = Kleisli[WeakOps, Array2D, Array2D] { value: Array2D =>
      StrongOps[Eval[INDArray], Eval[Option[INDArray]], DifferentiableINDArray.type](Eval.later(value.toNDArray), DifferentiableINDArray).erase[Array2D]
    }

    override def reciprocalDouble: WeakOps[(Double) => Double] = ???

    override def reciprocalArray2D: WeakOps[(Array2D) => Array2D] = ???

    override def negative: WeakOps[(Array2D) => Array2D] = ???

    override def multiplyArray2DArray2D: WeakOps[(Array2D, Array2D) => Array2D] = ???

    override def multiplyArray2DDouble: WeakOps[(Array2D, Double) => Array2D] = ???

    override def multiplyDoubleDouble: WeakOps[(Double, Double) => Double] = ???

    override def addArray2DArray2D: WeakOps[(Array2D, Array2D) => Array2D] = ???

    override def addArray2DDouble: WeakOps[(Array2D, Double) => Array2D] = ???

    override def addDoubleDouble: WeakOps[(Double, Double) => Double] = ???

    override def dot: WeakOps[(Array2D, Array2D) => Array2D] = ???

    override def max: WeakOps[(Array2D, Double) => Array2D] = ???

    override def exp: WeakOps[(Array2D) => Array2D] = ???

    override def sqrtDouble: WeakOps[(Double) => Double] = ???

    override def sqrtArray2D: WeakOps[(Array2D) => Array2D] = ???

    override def zeros(numberOfRows: Int, numberOfColumns: Int): WeakOps[Array2D] = ???

    override def randn(numberOfRows: Int, numberOfColumns: Int): WeakOps[Array2D] = ???
  }

}

import DeepLearning.{DeepLearningInstances => _, _}

@typeclass
trait DeepLearning[F[_]] extends PointfreeOperators[F] {

  def sigmoid[A](input: F[A])(implicit constrait: F[A] <:< F[Array2D]): F[Array2D] = {
    import DeepLearning.ops._
    implicit def self = this
    liftDouble(1.0) / (liftDouble(1.0) + exp(-constrait(input)))
  }

  def relu[A](input: F[A])(implicit constrait: F[A] <:< F[Array2D]) = {
    import DeepLearning.ops._
    implicit def self = this
    max(constrait(input), 0.0)
  }

  def fullyConnected[A](input: F[A], weight: F[Array2D], bias: F[Array2D])(implicit constrait: F[A] <:< F[Array2D]) = {
    import DeepLearning.ops._
    implicit def self = this
    dot(constrait(input), weight) + bias
  }

  def fullyConnectedThenRelu[A](input: F[A], inputSize: Int, outputSize: Int)(implicit constrait: F[A] <:< F[Array2D]) = {
    import DeepLearning.ops._
    implicit def self = this
    val weight = randn(inputSize, outputSize) / sqrtDouble(inputSize.toDouble / 2.0)
    val bias = zeros(1, outputSize)
    constrait(input).fullyConnected(weight, bias).relu
  }

}