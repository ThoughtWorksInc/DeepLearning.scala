package com.thoughtworks


import com.thoughtworks.DeepLearning.Differentiable.Immutable.ConstantDifference
import com.thoughtworks.DeepLearning.Differentiable.DifferentialbeINDArray.Delta
import com.thoughtworks.DeepLearning.Differentiable.DifferentialbeINDArray.Delta.{HasChange, NoChange}
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

    type Difference <: Semigroupable

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

    type Aux[Value, Difference0 <: Semigroupable, Self0] = Differentiable[Value] {
      type Difference = Difference0
      type Self = Self0
    }

    object DifferentialbeINDArray {

      sealed trait Delta extends Semigroupable {
        type Self = Delta
      }

      object Delta {

        case object NoChange extends Delta {
          override def append(other: Self): Self = other
        }

        final case class HasChange(delta: INDArray) extends Delta {
          override def append(other: Delta): Delta = other match {
            case NoChange => this
            case HasChange(otherChanges) => HasChange(delta + otherChanges)
          }
        }

      }

    }

    final case class DifferentialbeINDArray(value: INDArray) extends Differentiable[INDArray] {
      override type Difference = DifferentialbeINDArray.Delta

      override type Self = DifferentialbeINDArray

      override def applyPatch(patch: Delta, learningRate: Double): DifferentialbeINDArray = {
        patch match {
          case NoChange => this
          case HasChange(delta) => copy(value + delta * learningRate)
        }
      }
    }

    trait DifferentiableFunction[Input, Output] extends Differentiable[DifferentiableFunction[Input, Output]] {

      def forward[InputDifference <: Semigroupable, InputDifferentiable <: Differentiable.Aux[Input, InputDifference, InputDifferentiable]]
      (input: InputDifferentiable): DifferentiableFunction.Cache[Input, InputDifference, InputDifferentiable, Output, Self]

    }

    object DifferentiableFunction {


      trait Differences[Input, InputDifference <: Semigroupable, InputDifferentiable <: Differentiable.Aux[Input, InputDifference, InputDifferentiable], Weight] {

        type WeightDifference <: Semigroupable

        type WeightDifferentiable <: Differentiable.Aux[Weight, WeightDifference, WeightDifferentiable]

        def inputDifference: InputDifference

        def weightDifference: WeightDifference

      }

      trait Cache[Input, InputDifference <: Semigroupable, InputDifferentiable <: Differentiable.Aux[Input, InputDifference, InputDifferentiable], Output, Weight] {

        type OutputDifference <: Semigroupable

        type OutputDifferentiable <: Differentiable.Aux[Output, OutputDifference, OutputDifferentiable]

        def output: OutputDifferentiable

        def backward(difference: OutputDifference): Differences[Input, InputDifference, InputDifferentiable, Weight]

      }

      object Multiply extends DifferentiableFunction[INDArray, DifferentiableFunction[INDArray, INDArray]] with Immutable {

        override type Self = Multiply.type

        //
        //        override def forward(input: Differentiable[INDArray]) = new Cache[INDArray, DifferentiableFunction[INDArray, INDArray], Self] {
        //          override def output: OutputDifferentiable = ???
        //
        //          override def backward(difference: OutputDifference): Differences[INDArray, Self] = ???
        //        }
        //        override def forward[InputDifference <: Semigroupable, InputDifferentiable <: Aux[INDArray, InputDifference, InputDifferentiable]](input: InputDifferentiable) = {
        //          new Cache[INDArray, InputDifference, InputDifferentiable, DifferentiableFunction[INDArray, INDArray], Self] {
        ////            override type OutputDifference = this.type
        //
        //            override def output: OutputDifference = ???
        //
        //            override def backward(difference: this.type): Differences[INDArray, InputDifference, InputDifferentiable, Self] = ???
        //          }
        //        }
        final case class PartialAppliedFunction(input0: DifferentialbeINDArray)
          extends DifferentiableFunction[INDArray, INDArray] {
          override type Self = PartialAppliedFunction
          override type Difference = Delta

          override def forward[InputDifference <: Semigroupable, InputDifferentiable <: Aux[INDArray, InputDifference, InputDifferentiable]]
          (input1: InputDifferentiable) = new Cache[INDArray, InputDifference, InputDifferentiable, INDArray, Self] {

            override type OutputDifference = Delta

            override type OutputDifferentiable = DifferentialbeINDArray

            override def output: OutputDifferentiable = {
              input0 match {
                case DifferentialbeINDArray(data0) =>
                  input1 match {
                    case DifferentialbeINDArray(data1) =>
                      DifferentialbeINDArray(data0 + data1)
                  }
              }
            }

            override def backward(difference: OutputDifference) = ???
//            new Differences[INDArray, InputDifference, InputDifferentiable, PartialAppliedFunction[InputDifference, InputDifferentiable]] {
//
//              override def inputDifference: InputDifference = ???
//
//              override def weightDifference: WeightDifference = ???
//            }
          }

          override def applyPatch(patch: Difference, learningRate: Double): Self = {
            PartialAppliedFunction(input0.applyPatch(patch, learningRate))
          }
        }

        override def forward[InputDifference <: Semigroupable, InputDifferentiable <: Aux[INDArray, InputDifference, InputDifferentiable]]
        (input: InputDifferentiable) = new Cache[INDArray, InputDifference, InputDifferentiable, DifferentiableFunction[INDArray, INDArray], Self] {

//          override type OutputDifferentiable =

          override def output: OutputDifferentiable = ???

          override def backward(difference: OutputDifference): Differences[INDArray, InputDifference, InputDifferentiable, Multiply.type] = ???
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