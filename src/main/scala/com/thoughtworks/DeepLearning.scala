package com.thoughtworks


import cats.data.{Cokleisli, Kleisli}
import cats.kernel.std.DoubleGroup
import cats.{Applicative, Eval, Monoid}
import com.thoughtworks.DeepLearning.BinaryOperator.OperatorDispatcher
import com.thoughtworks.Differentiable.DifferentiableFunction.{AbstractDifferentiableFunction, BackwardPass, ForwardPass}
import com.thoughtworks.Differentiable.Pure.NoPatch
import com.thoughtworks.Differentiable.{DifferentiableFunction, _}
import com.thoughtworks.Pointfree.Ski
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import simulacrum.{op, typeclass}

import scala.language.{existentials, higherKinds, implicitConversions}

// Disable + for concatenation
import Predef.{any2stringadd => _, _}

@typeclass
trait DeepLearning[F[_]] {

  import DeepLearning.Array2D

  def liftDouble: Kleisli[F, Double, Double]

  def liftArray2D: Kleisli[F, Array2D, Array2D]

  def liftINDArray: Kleisli[F, INDArray, Array2D]

  def reciprocal: F[Array2D => Array2D]

  def negative: F[Array2D => Array2D]

  def mul: F[Array2D => Array2D => Array2D]

  def addArrayArray: F[Array2D => Array2D => Array2D]

  def addArrayDouble: F[Array2D => Double => Array2D]

  def addDoubleDouble: F[Double => Double => Double]

  def dot: F[Array2D => Array2D => Array2D]

  def max: F[Array2D => Double => Array2D]

  def exp: F[Array2D => Array2D]

}

object DeepLearning {

  type Array2D = Array[Array[Double]]


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
    val liftArray2D = Kleisli { value: Array2D =>
      DifferentiableINDArray(Eval.later(value.toNDArray))
    }
    val lift = Kleisli { value: INDArray =>
      DifferentiableINDArray(Eval.now(value))
    }

    val unlift = Cokleisli[Differentiable, Array2D, INDArray] {
      case DifferentiableINDArray(eval) => eval.value
    }

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

  final case class DifferentiableINDArray(override val data: Eval[_ <: INDArray]) extends Differentiable[Array2D] {
    override type Data = INDArray
    override type Delta = Option[INDArray]
    override val monoid: Eval[_ <: Monoid[Delta]] = DifferentiableINDArray.evalInstances
    override val patch: Eval[_ <: Patch[Data, Delta]] = DifferentiableINDArray.evalInstances
  }

  object Max extends DifferentiableFunction[Array2D, Double => Array2D] with Pure {
    override def forward[InputData, InputDelta] = {
      case DifferentiableINDArray(a) =>
        val partiallyAppled1 = new AbstractDifferentiableFunction(a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) with DifferentiableFunction[Double, Array2D] {
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


  object Negative extends DifferentiableFunction[Array2D, Array2D] with Pure {
    override def forward[InputData, InputDelta] = {
      case DifferentiableINDArray(a) =>
        val evalOutput = a.map(_.neg)
        ForwardPass(DifferentiableINDArray(evalOutput), { outputDifference: Eval[_ <: Option[INDArray]] =>
          BackwardPass(NoPatch.eval, outputDifference.map[Option[INDArray]](_.map(_.neg)))
        })
    }
  }

  object Reciprocal extends DifferentiableFunction[Array2D, Array2D] with Pure {
    override def forward[InputData, InputDelta] = {
      case DifferentiableINDArray(a) =>
        val evalOutput = a.map(_ rdiv 1)
        ForwardPass(DifferentiableINDArray(evalOutput), { outputDifference: Eval[_ <: Option[INDArray]] =>

          BackwardPass(NoPatch.eval, outputDifference.flatMap[Option[INDArray]] {
            case None =>
              Eval.now(None)
            case Some(outputDelta) =>
              a.map { aValue: INDArray =>
                Some(-outputDelta / (aValue * aValue))
              }
          })
        })
    }
  }


  object Exp extends DifferentiableFunction[Array2D, Array2D] with Pure {
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

  object AddArrayDouble extends DifferentiableFunction[Array2D, Double => Array2D] with Pure {
    override def forward[AData, ADelta] = {
      case DifferentiableINDArray(a) =>
        val partiallyAppled1 = new AbstractDifferentiableFunction(a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) with DifferentiableFunction[Double, Array2D] {
          override def forward[BData, BDelta] = {
            case DifferentiableDouble(b) =>
              val output = Applicative[Eval].map2(a, b) { (aData: INDArray, bData: Double) =>
                aData + bData
              }.memoize
              ForwardPass(DifferentiableINDArray(output), { outputDifference: Eval[_ <: Option[INDArray]] =>
                BackwardPass(outputDifference, outputDifference.map {
                  case None => 0
                  case Some(od) => od.sumT
                }.memoize)
              })
          }
        }
        ForwardPass(partiallyAppled1, { outputDifference: Eval[_ <: Option[INDArray]] =>
          BackwardPass(NoPatch.eval, outputDifference)
        })
    }
  }

  object AddArrayArray extends DifferentiableFunction[Array2D, Array2D => Array2D] with Pure {
    override def forward[AData, ADelta] = {
      case DifferentiableINDArray(a) =>
        val partiallyAppled1 = new AbstractDifferentiableFunction(a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) with DifferentiableFunction[Array2D, Array2D] {
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

  object Mul extends DifferentiableFunction[Array2D, Array2D => Array2D] with Pure {
    override def forward[AData, ADelta] = {
      case DifferentiableINDArray(a) =>
        val partiallyAppled1 = new AbstractDifferentiableFunction(a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) with DifferentiableFunction[Array2D, Array2D] {
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

  object Dot extends DifferentiableFunction[Array2D, Array2D => Array2D] with Pure {
    override def forward[AData, ADelta] = {
      case DifferentiableINDArray(a) =>
        val partiallyAppled1 = new AbstractDifferentiableFunction(a, DifferentiableINDArray.evalInstances, DifferentiableINDArray.evalInstances) with DifferentiableFunction[Array2D, Array2D] {
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
  trait PointfreeDeepLearning[F[_]] extends PointfreeFreezing[F] with DeepLearning[F] {

    def +[A, B](left: F[A], right: F[B])(implicit dispatcher: OperatorDispatcher[A, B]): F[dispatcher.Out] = {
      dispatcher(left, right)(new BinaryOperator[F] {
        override def doubleArray(left: F[Double], right: F[Array2D]): F[Array2D] = {
          implicit def self = PointfreeDeepLearning.this
          import PointfreeDeepLearning.ops._
          addArrayDouble ap right ap left
        }

        override def arrayArray(left: F[Array2D], right: F[Array2D]): F[Array2D] = {
          implicit def self = PointfreeDeepLearning.this
          import PointfreeDeepLearning.ops._
          addArrayArray ap left ap right
        }

        override def doubleDouble(left: F[Double], right: F[Double]): F[Double] = {
          implicit def self = PointfreeDeepLearning.this
          import PointfreeDeepLearning.ops._
          addDoubleDouble ap left ap right
        }

        override def arrayDouble(left: F[Array2D], right: F[Double]): F[Array2D] = {
          implicit def self = PointfreeDeepLearning.this
          import PointfreeDeepLearning.ops._
          addArrayDouble ap left ap right
        }
      })
    }

    def -[A](left: F[A], right: F[Array2D])(implicit constrait: F[A] <:< F[Array2D]) = {
      implicit def self = this
      import PointfreeDeepLearning.ops._
      addArrayArray ap constrait(left) ap (negative ap right)
    }

    def *[A](left: F[A], right: F[Array2D])(implicit constrait: F[A] <:< F[Array2D]) = {
      implicit def self = this
      import PointfreeDeepLearning.ops._
      mul ap constrait(left) ap right
    }

    def /[A](left: F[A], right: F[Array2D])(implicit constrait: F[A] <:< F[Array2D]) = {
      implicit def self = this
      import PointfreeDeepLearning.ops._
      mul ap constrait(left) ap (reciprocal ap right)
    }

    def unary_-[A](value: F[A])(implicit constrait: F[A] <:< F[Array2D]) = {
      implicit def self = this
      import PointfreeDeepLearning.ops._
      negative ap constrait(value)
    }

    def max[A](left: F[A], right: F[Double])(implicit constrait: F[A] <:< F[Array2D]): F[Array2D] = {
      implicit def self = this
      import PointfreeDeepLearning.ops._
      ap(max)(constrait(left)) ap right
    }

    def dot[A](left: F[A], right: F[Array2D])(implicit constrait: F[A] <:< F[Array2D]): F[Array2D] = {
      implicit def self = this
      import PointfreeDeepLearning.ops._
      ap(dot)(constrait(left)) ap right
    }

    def exp[A](value: F[A])(implicit constrait: F[A] <:< F[Array2D]): F[Array2D] = {
      implicit def self = this
      import PointfreeDeepLearning.ops._
      ap(exp)(constrait(value))
    }

    final def relu = {
      implicit def self = this
      import PointfreeDeepLearning.ops._
      flip[Array2D, Double, Array2D] ap max ap (freeze[Double] ap liftDouble(0.0))
    }

    final def fullyConnected(weight: F[Array2D], bias: F[Array2D]) = {
      implicit def self = this
      import PointfreeDeepLearning.ops._
      andThen[Array2D, Array2D, Array2D](flip[Array2D, Array2D, Array2D] ap dot ap weight, addArrayArray ap bias)
    }

    final def fullyConnectedThenRelu
    (inputSize: Int, outputSize: Int)
    : F[Array2D => Array2D] = {
      val fc: F[Array2D => Array2D] = fullyConnected(liftINDArray(Nd4j.randn(inputSize, outputSize) / math.sqrt(inputSize / 2)), liftINDArray(Nd4j.zeros(outputSize)))
      andThen[Array2D, Array2D, Array2D](fc, relu)
    }

    final def sigmoid(input: F[Array2D]): F[Array2D] = {
      import PointfreeDeepLearning.ops._
      implicit def self = this
      //      liftDouble(1) /
      //      liftDouble(1) + liftDouble(1)
      (liftDouble(1) + exp(-input))
      //      ???
    }

    final def sigmoidF(implicit liftDouble: Kleisli[F, Double, Double]): F[Array2D => Array2D] = {
      implicit def self = this
      PointfreeDeepLearning.withParameterInstances[F, Array2D](this).sigmoid(id[Array2D])
    }

  }


  object PointfreeDeepLearning {

    import PointfreeDeepLearning.ops._

    trait WithParameter[F[_], Parameter] extends PointfreeFreezing.WithParameter[F, Parameter] with PointfreeDeepLearning[Lambda[X => F[Parameter => X]]] {
      implicit protected def outer: PointfreeDeepLearning[F]

      override def liftDouble = kleisliWithParameter(outer, outer.liftDouble)

      override def liftArray2D = kleisliWithParameter(outer, outer.liftArray2D)

      override def liftINDArray = kleisliWithParameter(outer, outer.liftINDArray)

      override def mul = outer.mul.withParameter

      override def addArrayArray = outer.addArrayArray.withParameter

      override def addArrayDouble = outer.addArrayDouble.withParameter

      override def addDoubleDouble = outer.addDoubleDouble.withParameter

      override def negative = outer.negative.withParameter

      override def reciprocal = outer.reciprocal.withParameter

      override def max = (outer: DeepLearning[F]).max.withParameter

      override def exp = (outer: DeepLearning[F]).exp.withParameter

      override def dot = (outer: DeepLearning[F]).dot.withParameter
    }

    implicit def withParameterInstances[F[_], Parameter](implicit underlying: PointfreeDeepLearning[F]) = new WithParameter[F, Parameter] {
      override implicit protected def outer = underlying
    }

  }

  implicit object DeepLearningInstances extends PointfreeDeepLearning[Differentiable] with DifferentiableInstances {
    override def mul = Mul

    override def dot = Dot

    override def addArrayArray = AddArrayArray

    override def addArrayDouble = AddArrayDouble //: Differentiable[(INDArray) => (Double) => INDArray] = ???

    override def addDoubleDouble: Differentiable[(Double) => (Double) => Double] = ???

    override def max = Max

    override def exp = Exp

    override def negative = Negative

    override def reciprocal = Reciprocal

    override def liftDouble: Kleisli[Differentiable, Double, Double] = DifferentiableDouble.lift

    override def liftArray2D = DifferentiableINDArray.liftArray2D

    override def liftINDArray = DifferentiableINDArray.lift
  }

}

