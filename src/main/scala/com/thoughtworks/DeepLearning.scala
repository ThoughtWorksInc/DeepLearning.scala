package com.thoughtworks


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

  sealed trait Patch[Data, Difference0] extends Semigroup[Difference0] {
    type Difference = Difference0

    def applyPatch(weight: Data, patch: Difference0, learningRate: Double): Data
  }

  object NeverChange

  object Patch {

    implicit object INDArrayPatch extends Patch[INDArray, INDArray] {
      override def applyPatch(weight: INDArray, patch: INDArray, learningRate: Double): INDArray = {
        weight + patch * learningRate
      }

      override def append(f1: INDArray, f2: => INDArray): INDArray = {
        f1 + f2
      }
    }

    implicit def neverChangePatch[Data <: Singleton] = new Patch[Data, NeverChange.type] {
      override final def applyPatch(weight: Data, patch: Difference, learningRate: Double) = weight

      override final def append(f1: NeverChange.type, f2: => NeverChange.type) = NeverChange
    }
  }

  trait DifferentiableFunction[-Input, +Output] {

    type Self >: this.type <: DifferentiableFunction[Input, Output]

    type WeightDifference

    implicit def weightPatch: Patch[Self, WeightDifference]

    def forward(input: Input)(implicit expectedInputPatch: Patch[_ <: Input, _]): DifferentiableFunction.Cache[_ <: Output, expectedInputPatch.Difference, WeightDifference]

  }

  object DifferentiableFunction {

    trait Differences[InputDifference, WeightDifference] {
      outer =>

      def inputDifference: InputDifference

      def weightDifference: WeightDifference

    }

    trait Cache[Output, InputDifference, WeightDifference] {

      type OutputDifference

      implicit def outputPatch: Patch[Output, OutputDifference]

      def output: Output

      def backward(difference: OutputDifference): Differences[InputDifference, WeightDifference]

      final def unsafeCast[Output1, InputDifference1, WeightDifference1] = {
        asInstanceOf[Cache[Output1, InputDifference1, WeightDifference1]]
      }

    }

    type Aux[Input, Output, Self0] = DifferentiableFunction[Input, Output] {
      type Self = Self0
    }

    object PartialApplied {

      final case class PartialAppliedDifference[InputDifference, FDifference]
      (inputDifference: InputDifference, weightDifference: FDifference)
        extends Differences[InputDifference, FDifference]

      // TODO: Should implement it via shapeless.Generic
      //
      //      object PartialAppliedDifference {
      //
      //        implicit def partialAppliedPatch[Input0, Input1, Output, InputDifference0, FDifference, P <: PartialApplied[Input0, Input1, Output, InputDifference0, FDifference, F], F <: DifferentiableFunction.Aux[Input0, P, F] {
      //          type WeightDifference = FDifference
      //        }]
      //        (
      //          implicit inputPatch: Patch[Input0, InputDifference0],
      //          weightPatch: Patch[F, FDifference]
      //        ) = new Patch[PartialApplied[Input0, Input1, Output, InputDifference0, FDifference, F], PartialAppliedDifference[InputDifference0, FDifference]] {
      //          override def applyPatch(weight: PartialApplied[Input0, Input1, Output, InputDifference0, FDifference, F], patch: PartialAppliedDifference[InputDifference0, FDifference], learningRate: Double): PartialApplied[Input0, Input1, Output, InputDifference0, FDifference, F] = {
      //            new PartialApplied[Input0, Input1, Output, InputDifference0, FDifference, F](
      //              inputPatch.applyPatch(weight.input0, patch.inputDifference, learningRate),
      //              weightPatch.applyPatch(weight.f, patch.weightDifference, learningRate)
      //            )
      //          }
      //
      //          override def append(f1: PartialAppliedDifference[InputDifference0, FDifference], f2: => PartialAppliedDifference[InputDifference0, FDifference]): PartialAppliedDifference[InputDifference0, FDifference] = {
      //            new PartialAppliedDifference[InputDifference0, FDifference](
      //              inputPatch.append(f1.inputDifference, f2.inputDifference),
      //              weightPatch.append(f1.weightDifference, f2.weightDifference)
      //            )
      //
      //          }
      //        }
      //
      //      }

    }


    final case class PartialApplied[Input0, Input1, Output, InputDifference0, FDifference, F <: DifferentiableFunction.Aux[Input0, _, F] {
      type WeightDifference = FDifference
    }]
    (input0: Input0, f: F)
    (implicit inputPatch: Patch[Input0, InputDifference0])
      extends DifferentiableFunction[Input1, Output]
        with Cache[PartialApplied[Input0, Input1, Output, InputDifference0, FDifference, F], InputDifference0, FDifference] {

      type Self = PartialApplied[Input0, Input1, Output, InputDifference0, FDifference, F]

      type WeightDifference = PartialApplied.PartialAppliedDifference[InputDifference0, FDifference]

      override implicit def weightPatch = {
        new Patch[Self, WeightDifference] {
          override def applyPatch(weight: Self, patch: WeightDifference, learningRate: Double): Self = {
            new Self(
              inputPatch.applyPatch(weight.input0, patch.inputDifference, learningRate),
              weight.f.weightPatch.applyPatch(weight.f, patch.weightDifference, learningRate)
            )

          }

          override def append(f1: WeightDifference, f2: => WeightDifference): WeightDifference = {
            new WeightDifference(
              inputPatch.append(f1.inputDifference, f2.inputDifference),
              f.weightPatch.append(f1.weightDifference, f2.weightDifference)
            )
          }
        }
      }

      override def output: Self = this

      override def forward(input: Input1)(implicit inputPatch: Patch[_ <: Input1, _]): Cache[_ <: Output, inputPatch.Difference, WeightDifference] = ???

      type OutputDifference = WeightDifference

      override implicit def outputPatch: Patch[Self, OutputDifference] = weightPatch

      override def backward(difference: OutputDifference): WeightDifference = difference

    }


    trait PureFunction {
      _: DifferentiableFunction[_, _] with Singleton =>
      override type Self = this.type

      override type WeightDifference = NeverChange.type

      override implicit def weightPatch: Patch[Self, WeightDifference] = Patch.neverChangePatch
    }

    object Multiply extends DifferentiableFunction[INDArray, DifferentiableFunction[INDArray, INDArray]] with PureFunction {
      override def forward(input: INDArray)(implicit inputPatch: Patch[_ <: INDArray, _]): Cache[_ <: PartialApplied[INDArray, INDArray, INDArray, INDArray, NeverChange.type, Multiply.type], inputPatch.Difference, WeightDifference] = {
        inputPatch match {
          case Patch.INDArrayPatch =>
            PartialApplied[INDArray, INDArray, INDArray, INDArray, NeverChange.type, Multiply.type](input, Multiply.this)
          case _ =>
            throw new IllegalArgumentException(s"Unsupported patch type ${inputPatch}")
        }
      }.unsafeCast
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

