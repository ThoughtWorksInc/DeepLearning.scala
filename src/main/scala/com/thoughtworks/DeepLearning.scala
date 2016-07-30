package com.thoughtworks


import com.thoughtworks.DeepLearning.Differentiable.Immutable.ConstantDifference
import com.thoughtworks.DeepLearning.Differentiable.DifferentialbeINDArray.Delta
import com.thoughtworks.DeepLearning.Differentiable.DifferentialbeINDArray.Delta.{NonZeroDelta, ZeroDelta}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms._
import shapeless.{::, DepFn0, DepFn1, DepFn2, HList, HNil, Poly0, PolyApply, the}

import scala.language.existentials
import scala.language.higherKinds
import scalaz.{Apply, Arrow, Category, Choice, Compose, Semigroup, Split, Strong}

object DeepLearning {

  trait Substitution[=>:[_, _]] {
    def substitute[A, B, C](x: A =>: B =>: C, y: A =>: B): A =>: C
  }

  trait Constant[=>:[_, _]] {
    def constant[A, B, C](x: A =>: B): C =>: A =>: B
  }

  trait SKICombinator[=>:[_, _]] extends Substitution[=>:] with Constant[=>:] with Category[=>:]

  trait Multiply[=>:[_, _]] {
    def multiply: INDArray =>: INDArray =>: INDArray
  }

  trait Semigroupable {
    type Self >: this.type

    def append(other: Self): Self
  }

  object Semigroupable {

    type Aux[Self0] = Semigroupable {
      type Self = Self0
    }


    implicit def semigroupableSemigroup[A <: Aux[A]] = new Semigroup[A] {
      override def append(f1: A, f2: => A): A = f1.append(f2)
    }

  }

  trait Differentiable[Value] {

    type Self >: this.type

    type Difference <: Semigroupable.Aux[Difference]

    def applyPatch(patch: Difference, learningRate: Double): Self

  }

  object Differentiable {

    object Immutable {

      object ConstantDifference extends Semigroupable {
        override type Self = this.type

        override def append(other: Self): Self = this
      }

    }

    trait Immutable {
      _: Differentiable[_] =>
      final override type Difference = ConstantDifference.type

      final override def applyPatch(patch: Difference, learningRate: Double): Self = this
    }

    type Aux[Value, Difference0 <: Semigroupable.Aux[Difference0], Self0] = Differentiable[Value] {
      type Difference = Difference0
      type Self = Self0
    }

    object DifferentialbeINDArray {

      sealed trait Delta extends Semigroupable {
        type Self = Delta
      }

      object Delta {

        case object ZeroDelta extends Delta {
          override def append(other: Self): Self = other
        }

        final case class NonZeroDelta(delta: INDArray) extends Delta {
          override def append(other: Delta): Delta = other match {
            case ZeroDelta => this
            case NonZeroDelta(otherChanges) => NonZeroDelta(delta + otherChanges)
          }
        }

      }

    }

    final case class DifferentialbeINDArray(value: INDArray) extends Differentiable[INDArray] {
      override type Difference = DifferentialbeINDArray.Delta

      override type Self = DifferentialbeINDArray

      override def applyPatch(patch: Delta, learningRate: Double): DifferentialbeINDArray = {
        patch match {
          case ZeroDelta => this
          case NonZeroDelta(delta) => copy(value + delta * learningRate)
        }
      }
    }

    trait DifferentiableFunction[Input, Output] extends Differentiable[DifferentiableFunction[Input, Output]] {

      def forward[InputDifference <: Semigroupable.Aux[InputDifference], InputDifferentiable <: Differentiable.Aux[Input, InputDifference, InputDifferentiable]]
      (input: InputDifferentiable): DifferentiableFunction.Cache[Output, InputDifference, Difference]

    }

    object DifferentiableFunction {


      trait Differences[InputDifference <: Semigroupable.Aux[InputDifference], WeightDifference <: Semigroupable.Aux[WeightDifference]] {

        def inputDifference: InputDifference

        def weightDifference: WeightDifference

      }

      trait Cache[Output, InputDifference <: Semigroupable.Aux[InputDifference], WeightDifference <: Semigroupable.Aux[WeightDifference]] {

        type OutputDifference <: Semigroupable.Aux[OutputDifference]

        type OutputDifferentiable <: Differentiable.Aux[Output, OutputDifference, OutputDifferentiable]

        def output: OutputDifferentiable

        def backward(difference: OutputDifference): Differences[InputDifference, WeightDifference]

      }

      object Multiply extends DifferentiableFunction[INDArray, DifferentiableFunction[INDArray, INDArray]] with Immutable {
        outer =>

        override type Self = Multiply.type

        object PartialAppliedFunction {

          final case class PartialAppliedDifference[Difference0 <: Semigroupable.Aux[Difference0], Difference1 <: Semigroupable.Aux[Difference1]]
          (difference0: Difference0, difference1: Difference1) extends Semigroupable {
            override type Self = PartialAppliedDifference[Difference0, Difference1]

            override def append(other: Self): Self = new Self(difference0.append(other.difference0), difference1.append(other.difference1))
          }

        }

        final case class PartialAppliedFunction[
        InputDifference0 <: Semigroupable.Aux[InputDifference0],
        InputDifferentiable0 <: Aux[INDArray, InputDifference0, InputDifferentiable0]
        ]
        (input0: InputDifferentiable0)
          extends DifferentiableFunction[INDArray, INDArray] {
          override type Self = outer.Self#PartialAppliedFunction[InputDifference0, InputDifferentiable0]
          override type Difference = PartialAppliedFunction.PartialAppliedDifference[outer.Difference, InputDifference0]

          override def forward[InputDifference1 <: Semigroupable.Aux[InputDifference1], InputDifferentiable1 <: Aux[INDArray, InputDifference1, InputDifferentiable1]]
          (input1: InputDifferentiable1) = new Cache[INDArray, InputDifference1, Difference] {

            override type OutputDifference = Delta

            override type OutputDifferentiable = DifferentialbeINDArray

            override def output: OutputDifferentiable = {
              (input0: Aux[INDArray, _, _]) match {
                case DifferentialbeINDArray(data0) =>
                  input1 match {
                    case DifferentialbeINDArray(data1) =>
                      DifferentialbeINDArray(data0 + data1)
                  }
              }
            }

            override def backward(difference: OutputDifference) = ???

          }

          override def applyPatch(patch: Difference, learningRate: Double): Self = {
            val newOuter = outer.applyPatch(patch.difference0, learningRate)
            newOuter.PartialAppliedFunction[InputDifference0, InputDifferentiable0](input0.applyPatch(patch.difference1, learningRate))
          }
        }

        override def forward[InputDifference <: Semigroupable.Aux[InputDifference], InputDifferentiable <: Aux[INDArray, InputDifference, InputDifferentiable]]
        (input: InputDifferentiable) = new Cache[DifferentiableFunction[INDArray, INDArray], InputDifference, Difference] {

          //          override type OutputDifferentiable =

          override def output: OutputDifferentiable = ???

          override def backward(difference: OutputDifference): Differences[InputDifference, Difference] = ???
        }
      }

      implicit object DifferentiableFunctionInstances extends SKICombinator[DifferentiableFunction] with Multiply[DifferentiableFunction] {

        override def multiply: DifferentiableFunction[INDArray, DifferentiableFunction[INDArray, INDArray]] = Multiply

        override def compose[A, B, C](f: DifferentiableFunction[B, C], g: DifferentiableFunction[A, B]): DifferentiableFunction[A, C] = ???

        override def id[A]: DifferentiableFunction[A, A] = ???

        override def constant[A, B, C](x: DifferentiableFunction[A, B]): DifferentiableFunction[C, DifferentiableFunction[A, B]] = ???

        override def substitute[A, B, C](x: DifferentiableFunction[A, DifferentiableFunction[B, C]], y: DifferentiableFunction[A, B]): DifferentiableFunction[A, C] = ???
      }

    }

  }

}