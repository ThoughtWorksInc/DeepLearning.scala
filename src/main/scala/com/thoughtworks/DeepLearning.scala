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
import shapeless._
import simulacrum.{op, typeclass}
import com.dongxiguo.fastring.Fastring.Implicits._

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

    object DifferentiableDoubleInstances extends EvalMonoid[Double] with ToFastring[Eval[Double]] with Patch[Eval[Double], Eval[Double]] {


      override implicit def algebra = cats.std.double.doubleGroup

      override def apply(data: Eval[Double], delta: Eval[Double], learningRate: Double): Eval[Double] = {
        Applicative[Eval].map2(data, delta)(_ - _ * learningRate).memoize
      }

      override def apply(data: Eval[Double]) = {
        fast"${data.value}"
      }
    }

    override type Delta = Eval[Double]

    override def monoid = DifferentiableDoubleInstances

    override def patch = DifferentiableDoubleInstances

    override def toFastring = DifferentiableDoubleInstances
  }

  case object DifferentiableINDArray extends Differentiable[Eval[INDArray]] {

    override type Delta = Eval[Option[INDArray]]

    override def monoid = DifferentiableINDArrayInstances

    override def patch = DifferentiableINDArrayInstances

    override def toFastring = DifferentiableINDArrayInstances

    object DifferentiableINDArrayInstances extends Patch[Eval[INDArray], Eval[Option[INDArray]]] with EvalMonoid[Option[INDArray]] with ToFastring[Eval[INDArray]] {

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
          case Some(deltaValue) => weight.map(_ - deltaValue * learningRate)
        }.memoize
      }

      override def apply(data: Eval[INDArray]) = {
        fast"${data.value}"
      }
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

    implicit object Array2DINDArrayDispatcher extends OperatorDispatcher[Array2D, Array2D] {
      override type Out = Array2D

      override def apply[F[_] : BinaryOperator](left: F[Array2D], right: F[Array2D]): F[Array2D] = {
        BinaryOperator[F].arrayArray(left, right)
      }
    }

    implicit object DoubleDoubleDispatcher extends OperatorDispatcher[Double, Double] {
      override type Out = Double

      override def apply[F[_] : BinaryOperator](left: F[Double], right: F[Double]): F[Double] = {
        BinaryOperator[F].doubleDouble(left, right)
      }
    }

    implicit object DoubleINDArrayDispatcher extends OperatorDispatcher[Double, Array2D] {
      override type Out = Array2D

      override def apply[F[_] : BinaryOperator](left: F[Double], right: F[Array2D]): F[Array2D] = {
        BinaryOperator[F].doubleArray(left, right)
      }
    }

    implicit object Array2DDoubleDispatcher extends OperatorDispatcher[Array2D, Double] {
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

    def zeros(numberOfColumns: Int): F[Array2D] = zeros(1, numberOfColumns)

    def randn(numberOfRows: Int, numberOfColumns: Int): F[Array2D]

    protected def reciprocalDouble: F[Double => Double]

    protected def reciprocalArray2D: F[Array2D => Array2D]

    protected def negativeDouble: F[Double => Double]

    protected def negativeArray2D: F[Array2D => Array2D]

    protected def multiplyArray2DArray2D: F[(Array2D, Array2D) => Array2D]

    protected def multiplyArray2DDouble: F[(Array2D, Double) => Array2D]

    protected def multiplyDoubleDouble: F[(Double, Double) => Double]

    protected def addArray2DArray2D: F[(Array2D, Array2D) => Array2D]

    protected def addArray2DDouble: F[(Array2D, Double) => Array2D]

    protected def addDoubleDouble: F[(Double, Double) => Double]

    def dot: F[(Array2D, Array2D) => Array2D]

    protected def maxArray2DDouble: F[(Array2D, Double) => Array2D]

    protected def exponentialArray2D: F[Array2D => Array2D]

    protected def logArray2D: F[Array2D => Array2D]

    def broadcast(numberOfRows: Int, numberOfColumns: Int): F[Array2D => Array2D]

    def sum(dimensions: Int*): F[Array2D => Array2D]

    def reduceSum: F[Array2D => Double]

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


    def -[A, B](left: F[A], right: F[B])(implicit dispatcher: OperatorDispatcher[A, B]): F[dispatcher.Out] = {
      dispatcher(left, right)(new BinaryOperator[F] {
        override def doubleArray(left: F[Double], right: F[Array2D]): F[Array2D] = {
          addArray2DDouble(negativeArray2D(right), left)
        }

        override def arrayArray(left: F[Array2D], right: F[Array2D]): F[Array2D] = {
          addArray2DArray2D(left, negativeArray2D(right))
        }

        override def doubleDouble(left: F[Double], right: F[Double]): F[Double] = {
          addDoubleDouble(left, negativeDouble(right))
        }

        override def arrayDouble(left: F[Array2D], right: F[Double]): F[Array2D] = {
          addArray2DDouble(left, negativeDouble(right))
        }
      })
    }

    def *[A, B](left: F[A], right: F[B])(implicit dispatcher: OperatorDispatcher[A, B]): F[dispatcher.Out] = {
      dispatcher(left, right)(new BinaryOperator[F] {
        override def doubleArray(left: F[Double], right: F[Array2D]): F[Array2D] = {
          multiplyArray2DDouble(right, left)
        }

        override def arrayArray(left: F[Array2D], right: F[Array2D]): F[Array2D] = {
          multiplyArray2DArray2D(left, right)
        }

        override def doubleDouble(left: F[Double], right: F[Double]): F[Double] = {
          multiplyDoubleDouble(left, right)
        }

        override def arrayDouble(left: F[Array2D], right: F[Double]): F[Array2D] = {
          multiplyArray2DDouble(left, right)
        }
      })
    }

    def /[A, B](left: F[A], right: F[B])(implicit dispatcher: OperatorDispatcher[A, B]): F[dispatcher.Out] = {
      dispatcher(left, right)(new BinaryOperator[F] {
        override def doubleArray(left: F[Double], right: F[Array2D]): F[Array2D] = {
          multiplyArray2DDouble(reciprocalArray2D(right), left)
        }

        override def arrayArray(left: F[Array2D], right: F[Array2D]): F[Array2D] = {
          multiplyArray2DArray2D(left, reciprocalArray2D(right))
        }

        override def doubleDouble(left: F[Double], right: F[Double]): F[Double] = {
          multiplyDoubleDouble(left, reciprocalDouble(right))
        }

        override def arrayDouble(left: F[Array2D], right: F[Double]): F[Array2D] = {
          multiplyArray2DDouble(left, reciprocalDouble(right))
        }
      })
    }

    def max[A, B](left: F[A], right: F[B])(implicit dispatcher: OperatorDispatcher[A, B]): F[dispatcher.Out] = {
      dispatcher(left, right)(new BinaryOperator[F] {
        override def doubleArray(left: F[Double], right: F[Array2D]): F[Array2D] = {
          maxArray2DDouble(right, left)
        }

        override def arrayArray(left: F[Array2D], right: F[Array2D]): F[Array2D] = {
          ???
        }

        override def doubleDouble(left: F[Double], right: F[Double]): F[Double] = {
          ???
        }

        override def arrayDouble(left: F[Array2D], right: F[Double]): F[Array2D] = {
          maxArray2DDouble(left, right)
        }
      })
    }

    def unary_-[A](value: F[A])(implicit constrait: F[A] <:< F[Array2D]) = {
      negativeArray2D(constrait(value))
    }


    def exp[A](value: F[A])(implicit constrait: F[A] <:< F[Array2D]) = {
      exponentialArray2D(constrait(value))
    }

    def log[A](value: F[A])(implicit constrait: F[A] <:< F[Array2D]) = {
      logArray2D(constrait(value))
    }


  }

  object PointfreeOperators {

    import PointfreeOperators.ops._

    trait WithParameter[F[_], Parameter] extends PointfreeFreezing.WithParameter[F, Parameter] with PointfreeOperators[Lambda[X => F[Parameter => X]]] {
      implicit protected def outer: PointfreeOperators[F]

      override def liftDouble = kleisliWithParameter(outer, outer.liftDouble)

      override def liftArray2D = kleisliWithParameter(outer, outer.liftArray2D)

      override def broadcast(numberOfRows: Int, numberOfColumns: Int) = outer.broadcast(numberOfRows, numberOfColumns).withParameter

      override def reduceSum = outer.reduceSum.withParameter

      override def sum(dimensions: Int*) = outer.sum(dimensions: _*).withParameter

      protected override def multiplyArray2DArray2D = outer.multiplyArray2DArray2D.withParameter

      protected override def multiplyArray2DDouble = outer.multiplyArray2DDouble.withParameter

      protected override def multiplyDoubleDouble = outer.multiplyDoubleDouble.withParameter

      protected override def addArray2DArray2D = outer.addArray2DArray2D.withParameter

      protected override def addArray2DDouble = outer.addArray2DDouble.withParameter

      protected override def addDoubleDouble = outer.addDoubleDouble.withParameter

      protected override def negativeArray2D = outer.negativeArray2D.withParameter

      protected override def negativeDouble = outer.negativeDouble.withParameter

      protected override def reciprocalArray2D = outer.reciprocalArray2D.withParameter

      protected override def reciprocalDouble = outer.reciprocalDouble.withParameter

      protected override def maxArray2DDouble = outer.maxArray2DDouble.withParameter

      protected override def exponentialArray2D = outer.exponentialArray2D.withParameter

      protected override def logArray2D = outer.logArray2D.withParameter

      override def dot = (outer: PointfreeOperators[F]).dot.withParameter

      override def zeros(numberOfRows: Int, numberOfColumns: Int): F[Parameter => Array2D] = outer.zeros(numberOfRows, numberOfColumns).withParameter

      override def randn(numberOfRows: Int, numberOfColumns: Int): F[Parameter => Array2D] = outer.randn(numberOfRows, numberOfColumns).withParameter
    }

    implicit def withParameterInstances[F[_], Parameter](implicit underlying: PointfreeOperators[F]) = new WithParameter[F, Parameter] {
      override implicit protected def outer = underlying
    }

  }

  implicit object DeepLearningInstances extends DeepLearning[WeakOps] with DifferentiableInstances {

    override val liftDouble = Kleisli[WeakOps, Double, Double] { value: Double =>
      new WeakOps[Double] with Differentiable.AllOps[Eval[Double]] {
        override val self = Eval.now(value)
        override val typeClassInstance = DifferentiableDouble
      }
    }

    override val liftArray2D = Kleisli[WeakOps, Array2D, Array2D] { value: Array2D =>
      new WeakOps[Array2D] with Differentiable.AllOps[Eval[INDArray]] {
        override val self = Eval.later(value.toNDArray)
        override val typeClassInstance = DifferentiableINDArray
      }
    }

    override def sum(dimensions: Int*) = {
      DeepLearning.sum(dimensions: _*).pureOps[Array2D => Array2D]
    }

    override def reduceSum = {
      ReduceSum.pureOps[Array2D => Double]
    }

    override def broadcast(numberOfRows: Int, numberOfColumns: Int) = {
      DeepLearning.broadcast(numberOfRows, numberOfColumns).pureOps[Array2D => Array2D]
    }

    protected override def reciprocalDouble = {
      ReciprocalDouble.pureOps[Double => Double]
    }

    protected override def negativeDouble = {
      NegativeDouble.pureOps[Double => Double]
    }

    protected override def reciprocalArray2D = {
      ReciprocalINDArray.pureOps[Array2D => Array2D]

    }

    protected override def negativeArray2D = {
      NegativeINDArray.pureOps[Array2D => Array2D]
    }

    protected override def multiplyArray2DArray2D = {
      MultiplyINDArrayINDArray.pureOps[(Array2D, Array2D) => Array2D]
    }

    protected override def multiplyArray2DDouble = {
      MultiplyINDArrayDouble.pureOps[(Array2D, Double) => Array2D]
    }

    protected override def multiplyDoubleDouble = {
      MultiplyDoubleDouble.pureOps[(Double, Double) => Double]
    }

    protected override def addArray2DArray2D = {
      AddINDArrayINDArray.pureOps[(Array2D, Array2D) => Array2D]
    }

    protected override def addArray2DDouble = {
      AddINDArrayDouble.pureOps[(Array2D, Double) => Array2D]
    }

    protected override def addDoubleDouble = {
      AddDoubleDouble.pureOps[(Double, Double) => Double]
    }

    override def dot = {
      Dot.pureOps[(Array2D, Array2D) => Array2D]
    }

    override def maxArray2DDouble = {
      MaxINDArrayDouble.pureOps[(Array2D, Double) => Array2D]
    }

    override def exponentialArray2D = {
      ExponentialINDArray.pureOps[Array2D => Array2D]
    }

    override def logArray2D = {
      LogINDArray.pureOps[Array2D => Array2D]
    }

    override def zeros(numberOfRows: Int, numberOfColumns: Int) = {
      new WeakOps[Array2D] with Differentiable.AllOps[Eval[INDArray]] {
        override val self = Eval.later(Nd4j.zeros(numberOfRows, numberOfColumns))
        override val typeClassInstance = DifferentiableINDArray
      }
    }

    override def randn(numberOfRows: Int, numberOfColumns: Int) = {
      new WeakOps[Array2D] with Differentiable.AllOps[Eval[INDArray]] {
        override val self = Eval.later(Nd4j.randn(numberOfRows, numberOfColumns))
        override val typeClassInstance = DifferentiableINDArray
      }
    }
  }

  trait WithParameter[F[_], Parameter] extends PointfreeOperators.WithParameter[F, Parameter] with DeepLearning[Lambda[X => F[Parameter => X]]]

  implicit def withParameterInstances[F[_], Parameter](implicit underlying: DeepLearning[F]) = new WithParameter[F, Parameter] {
    override implicit protected def outer = underlying
  }

  val MaxINDArrayDouble = {
    DifferentiableFunction(
      HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], hlist: Eval[INDArray] :: Eval[Double] :: HNil, _: DifferentiableINDArrayDouble) =>
        val (a: Eval[INDArray]) :: (b: Eval[Double]) :: HNil = hlist
        ForwardPass(Applicative[Eval].map2(a, b)(Transforms.max).memoize, DifferentiableINDArray, { outputDelta: Eval[Option[INDArray]] =>
          val aDelta = outputDelta.flatMap[Option[INDArray]] {
            case None => Eval.now(None)
            case Some(outputDeltaValue) =>
              Applicative[Eval].map2(a, b) { (aData: INDArray, bData: Double) =>
                Some((aData gt bData) * outputDeltaValue)
              }
          }
          val bDelta = outputDelta.flatMap[Double] {
            case None => Eval.now(0)
            case Some(outputDeltaValue) =>
              Applicative[Eval].map2(a, b) { (aData: INDArray, bData: Double) =>
                ((aData lt bData) * outputDeltaValue).sumT
              }
          }
          BackwardPass(HNil: HNil, aDelta :: bDelta :: HNil)
        })
      }
    )
  }


  val NegativeINDArray = {
    DifferentiableFunction(HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], a: Eval[INDArray], _: DifferentiableINDArray.type) =>
      ForwardPass(a.map(_.neg), DifferentiableINDArray, { outputDelta: Eval[Option[INDArray]] =>
        BackwardPass(HNil: HNil, outputDelta.map(_.map(_.neg)))
      })
    })
  }

  val LogINDArray = {
    DifferentiableFunction(HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], a: Eval[INDArray], _: DifferentiableINDArray.type) =>
      ForwardPass(a.map(Transforms.log), DifferentiableINDArray, { outputDelta: Eval[Option[INDArray]] =>
        BackwardPass(
          HNil: HNil,
          outputDelta.flatMap[Option[INDArray]] {
            case None =>
              Eval.now(None)
            case Some(outputDeltaValue) => a.map { aData: INDArray =>
              Some(outputDeltaValue / aData)
            }
          }
        )
      })
    })
  }

  val ExponentialINDArray = {
    DifferentiableFunction(HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], a: Eval[INDArray], _: DifferentiableINDArray.type) =>
      val output = a.map(Transforms.exp)
      ForwardPass(output, DifferentiableINDArray, { outputDelta: Eval[Option[INDArray]] =>
        BackwardPass(
          HNil: HNil,
          outputDelta.flatMap[Option[INDArray]] {
            case None =>
              Eval.now(None)
            case Some(outputDeltaValue) => output.map { outputValue: INDArray =>
              Some(outputValue * outputDeltaValue)
            }
          }
        )
      })
    })
  }

  val ReciprocalINDArray = {
    DifferentiableFunction(HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], a: Eval[INDArray], _: DifferentiableINDArray.type) =>
      ForwardPass(a.map(_ rdiv 1).memoize, DifferentiableINDArray, { outputDelta: Eval[Option[INDArray]] =>
        BackwardPass(
          HNil: HNil,
          outputDelta.flatMap[Option[INDArray]] {
            case None => Eval.now(None)
            case Some(outputDeltaValue) =>
              a.map { aValue: INDArray =>
                Some(-outputDeltaValue / (aValue * aValue))
              }
          }.memoize
        )
      })
    })
  }

  val ReciprocalDouble = {
    DifferentiableFunction(HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], a: Eval[Double], _: DifferentiableDouble.type) =>
      ForwardPass(a.map(1.0 / _), DifferentiableDouble, { outputDelta: Eval[Double] =>
        BackwardPass(
          HNil: HNil,
          Applicative[Eval].map2(outputDelta, a) { (outputDeltaValue: Double, aValue: Double) =>
            -outputDeltaValue / (aValue * aValue)
          }
        )
      })
    })
  }

  val NegativeDouble = {
    DifferentiableFunction(HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], a: Eval[Double], _: DifferentiableDouble.type) =>
      ForwardPass(a.map(-_), DifferentiableDouble, { outputDelta: Eval[Double] =>
        BackwardPass(
          HNil: HNil,
          outputDelta.map(-_)
        )
      })
    })
  }

  val ReduceSum = {
    DifferentiableFunction(HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], a: Eval[INDArray], _: DifferentiableINDArray.type) =>
      val aShape = a.map(_.shape)
      ForwardPass(a.map(_.sumT), DifferentiableDouble, { outputDelta: Eval[Double] =>
        BackwardPass(
          HNil: HNil,
          Applicative[Eval].map2(aShape, outputDelta) { (aShapeValue, outputDeltaValue) =>
            if (outputDeltaValue == 0) {
              None
            } else {
              Some(Nd4j.valueArrayOf(aShapeValue, outputDeltaValue))
            }
          }
        )
      })
    })
  }

  def sum(dimensions: Int*) = {
    DifferentiableFunction(HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], a: Eval[INDArray], _: DifferentiableINDArray.type) =>
      val aShape = a.map(_.shape)
      ForwardPass(a.map(_.sum(dimensions: _*)), DifferentiableINDArray, { outputDelta: Eval[Option[INDArray]] =>
        BackwardPass(
          HNil: HNil,
          outputDelta.flatMap[Option[INDArray]] {
            case None => Eval.now(None)
            case Some(outputDeltaValue) =>
              aShape.map { aShapeValue =>
                Some(outputDeltaValue.broadcast(aShapeValue: _*))
              }
          }.memoize
        )
      })
    })
  }

  def broadcast(rows: Int, columns: Int) = {
    DifferentiableFunction(HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], a: Eval[INDArray], _: DifferentiableINDArray.type) =>
      val aShape = a.map(_.shape)
      ForwardPass(a.map(_.broadcast(rows, columns)), DifferentiableINDArray, { outputDelta: Eval[Option[INDArray]] =>
        BackwardPass(
          HNil: HNil,
          outputDelta.flatMap[Option[INDArray]] {
            case None => Eval.now(None)
            case Some(outputDeltaValue) =>
              aShape.map {
                case Array(1, 1) => Some(outputDeltaValue.sum(0, 1))
                case Array(1, _) => Some(outputDeltaValue.sum(0))
                case Array(_, 1) => Some(outputDeltaValue.sum(1))
                case Array(_, _) => Some(outputDeltaValue)
              }
          }.memoize
        )
      })
    })
  }

  private type DifferentiableINDArrayINDArray = DifferentiableHCons[
    Eval[INDArray], Eval[Option[INDArray]], DifferentiableINDArray.type,
    Eval[INDArray] :: HNil, Eval[Option[INDArray]] :: HNil, DifferentiableHCons[
    Eval[INDArray], Eval[Option[INDArray]], DifferentiableINDArray.type,
    HNil, HNil, DifferentiableHNil.type]]

  private type DifferentiableINDArrayDouble = DifferentiableHCons[
    Eval[INDArray], Eval[Option[INDArray]], DifferentiableINDArray.type,
    Eval[Double] :: HNil, Eval[Double] :: HNil, DifferentiableHCons[
    Eval[Double], Eval[Double], DifferentiableDouble.type,
    HNil, HNil, DifferentiableHNil.type]]

  private type DifferentiableDoubleDouble = DifferentiableHCons[
    Eval[Double], Eval[Double], DifferentiableDouble.type,
    Eval[Double] :: HNil, Eval[Double] :: HNil, DifferentiableHCons[
    Eval[Double], Eval[Double], DifferentiableDouble.type,
    HNil, HNil, DifferentiableHNil.type]]

  def sumAs(outputDeltaValue: INDArray, shape: Array[Int]) = shape match {
    case Array(1, 1) => outputDeltaValue.sum(0, 1)
    case Array(_, 1) => outputDeltaValue.sum(1)
    case Array(1, _) => outputDeltaValue.sum(0)
    case Array(_, _) => outputDeltaValue
  }

  val AddINDArrayINDArray = {
    DifferentiableFunction(
      HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], hlist: Eval[INDArray] :: Eval[INDArray] :: HNil, _: DifferentiableINDArrayINDArray) =>
        val (a: Eval[INDArray]) :: (b: Eval[INDArray]) :: HNil = hlist
        val aShape = a.map(_.shape).memoize
        val bShape = b.map(_.shape).memoize
        val newShape = Applicative[Eval].map2(aShape, bShape) {
          case (Array(aRows, aColumns), Array(bRows, bColumns)) =>
            Array(math.max(aRows, bRows), math.max(aColumns, bColumns))
        }
        ForwardPass(Applicative[Eval].map3(a, b, newShape) { (aValue, bValue, newShapeValue) =>
          aValue.broadcast(newShapeValue: _*) + bValue.broadcast(newShapeValue: _*)
        }.memoize, DifferentiableINDArray, { outputDelta: Eval[Option[INDArray]] =>
          BackwardPass(
            HNil: HNil,
            Applicative[Eval].map2(outputDelta, aShape) { (outputDeltaOption, shape) =>
              outputDeltaOption.map(sumAs(_, shape))
            } :: Applicative[Eval].map2(outputDelta, bShape) { (outputDeltaOption, shape) =>
              outputDeltaOption.map(sumAs(_, shape))
            } :: HNil)
        })
      }
    )
  }

  val AddINDArrayDouble = {
    DifferentiableFunction(
      HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], hlist: Eval[INDArray] :: Eval[Double] :: HNil, _: DifferentiableINDArrayDouble) =>
        val (a: Eval[INDArray]) :: (b: Eval[Double]) :: HNil = hlist
        ForwardPass(Applicative[Eval].map2(a, b)(_ add _).memoize, DifferentiableINDArray, { outputDelta: Eval[Option[INDArray]] =>
          val aDelta = outputDelta
          val bDelta = outputDelta.map {
            case None => 0.0
            case Some(outputDeltaValue) => outputDeltaValue.sumT
          }.memoize
          BackwardPass(HNil: HNil, aDelta :: bDelta :: HNil)
        })
      }
    )
  }

  val AddDoubleDouble = {
    DifferentiableFunction(
      HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], hlist: Eval[Double] :: Eval[Double] :: HNil, _: DifferentiableDoubleDouble) =>
        val (a: Eval[Double]) :: (b: Eval[Double]) :: HNil = hlist
        ForwardPass(Applicative[Eval].map2(a, b)(_ + _), DifferentiableDouble, { outputDelta: Eval[Double] =>
          BackwardPass(HNil: HNil, outputDelta :: outputDelta :: HNil)
        })
      }
    )
  }

  val MultiplyINDArrayINDArray = {
    DifferentiableFunction(
      HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], hlist: Eval[INDArray] :: Eval[INDArray] :: HNil, _: DifferentiableINDArrayINDArray) =>
        val (a: Eval[INDArray]) :: (b: Eval[INDArray]) :: HNil = hlist
        val aShape = a.map(_.shape)
        val bShape = b.map(_.shape)
        val newShape = Applicative[Eval].map2(aShape, bShape) {
          case (Array(aRows, aColumns), Array(bRows, bColumns)) =>
            Array(math.max(aRows, bRows), math.max(aColumns, bColumns))
        }
        ForwardPass(Applicative[Eval].map3(a, b, newShape) {
          (aValue, bValue, newShapeValue) =>
            aValue.broadcast(newShapeValue: _*) * bValue.broadcast(newShapeValue: _*)
        }.memoize, DifferentiableINDArray, { outputDelta: Eval[Option[INDArray]] =>
          BackwardPass(
            HNil: HNil,
            outputDelta.flatMap[Option[INDArray]] {
              case None => Eval.now(None)
              case Some(outputDeltaValue) =>
                Applicative[Eval].map2(b, aShape) { (bData: INDArray, aShapeValue) =>
                  Some(sumAs(bData.broadcast(outputDeltaValue.shape: _*) * outputDeltaValue, aShapeValue))
                }
            } ::
              outputDelta.flatMap[Option[INDArray]] {
                case None => Eval.now(None)
                case Some(outputDeltaValue) =>
                  Applicative[Eval].map2(a, bShape) { (aData: INDArray, bShapeValue) =>
                    Some(sumAs(aData.broadcast(outputDeltaValue.shape: _*) * outputDeltaValue, bShapeValue))
                  }
              } :: HNil
          )
        })
      }
    )
  }

  val MultiplyINDArrayDouble = {
    DifferentiableFunction(
      HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], hlist: Eval[INDArray] :: Eval[Double] :: HNil, _: DifferentiableINDArrayDouble) =>
        val (a: Eval[INDArray]) :: (b: Eval[Double]) :: HNil = hlist
        ForwardPass(Applicative[Eval].map2(a, b)(_ mul _).memoize, DifferentiableINDArray, { outputDelta: Eval[Option[INDArray]] =>
          val aDelta = outputDelta.flatMap[Option[INDArray]] {
            case None => Eval.now(None)
            case Some(outputDeltaValue) =>
              b.map { bData: Double =>
                Some(outputDeltaValue * bData)
              }
          }.memoize
          val bDelta = outputDelta.flatMap[Double] {
            case None => Eval.now(0.0)
            case Some(outputDeltaValue) =>
              a.map { aData: INDArray =>
                (aData * outputDeltaValue).sumT
              }
          }.memoize
          BackwardPass(HNil: HNil, aDelta :: bDelta :: HNil)
        })
      }
    )
  }

  val MultiplyDoubleDouble = {
    DifferentiableFunction(
      HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], hlist: Eval[Double] :: Eval[Double] :: HNil, _: DifferentiableDoubleDouble) =>
        val (a: Eval[Double]) :: (b: Eval[Double]) :: HNil = hlist
        ForwardPass(Applicative[Eval].map2(a, b)(_ * _), DifferentiableDouble, { outputDelta: Eval[Double] =>
          val aDelta = Applicative[Eval].map2(outputDelta, b)(_ * _)
          val bDelta = Applicative[Eval].map2(outputDelta, a)(_ * _)
          BackwardPass(HNil: HNil, aDelta :: bDelta :: HNil)
        })
      }
    )
  }

  val Dot = {
    DifferentiableFunction(
      HNilMonoid, HNilPatch, HNilToFastring, { (weight: HNil, monoid: Monoid[HNil], patch: Patch[HNil, HNil], toFastring: ToFastring[HNil], hlist: Eval[INDArray] :: Eval[INDArray] :: HNil, _: DifferentiableINDArrayINDArray) =>
        val (a: Eval[INDArray]) :: (b: Eval[INDArray]) :: HNil = hlist
        ForwardPass(Applicative[Eval].map2(a, b)(_ dot _).memoize, DifferentiableINDArray, { outputDelta: Eval[Option[INDArray]] =>
          BackwardPass(
            HNil: HNil,
            outputDelta.flatMap[Option[INDArray]] {
              case None => Eval.now(None)
              case Some(outputDeltaValue) =>
                b.map { bData =>
                  Some(outputDeltaValue.dot(bData.T))
                }
            } :: outputDelta.flatMap[Option[INDArray]] {
              case None => Eval.now(None)
              case Some(outputDeltaValue) =>
                a.map { aData =>
                  Some(aData.T.dot(outputDeltaValue))
                }
            } :: HNil
          )
        })
      }
    )
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
    dot(constrait(input), weight) + bias // 此处会自动 broardcast bias
  }

  def fullyConnectedThenRelu[A](input: F[A], inputSize: Int, outputSize: Int)(implicit constrait: F[A] <:< F[Array2D]) = {
    import DeepLearning.ops._
    implicit def self = this
    val weight = randn(inputSize, outputSize) / math.sqrt(inputSize.toDouble / 2.0)
    val bias = zeros(outputSize)
    constrait(input).fullyConnected(weight, bias).relu
  }

}
