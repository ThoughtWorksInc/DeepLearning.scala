package com.thoughtworks


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms._
import shapeless.{::, DepFn0, DepFn1, DepFn2, Generic, HList, HNil, Poly0, PolyApply, Widen, the}

import scala.language.existentials
import scala.language.higherKinds
import scalaz.syntax.Ops
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

  sealed trait Patch[Data, Difference] extends Semigroup[Difference] {
    def applyPatch(weight: Data, patch: Difference, learningRate: Double): Data
  }

  sealed trait Differentiable {

    type Difference

    type Self

    def self: Self

    implicit def patch: Patch[Self, Difference]

  }

  object Differentiable {
    type Aux[Data, Difference0] = Differentiable {
      type Self = Data
      type Difference = Difference0
    }
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

    final case class NeverChangePatch[Data <: Singleton]() extends Patch[Data, NeverChange.type] {
      override def applyPatch(weight: Data, patch: NeverChange.type, learningRate: Double) = weight

      override def append(f1: NeverChange.type, f2: => NeverChange.type) = NeverChange
    }

    implicit def neverChangePatch[Data <: Singleton] = new NeverChangePatch[Data]

    implicit object HNilPatch extends Patch[HNil, HNil] {
      override def applyPatch(weight: HNil, patch: HNil, learningRate: Double) = HNil

      override def append(f1: HNil, f2: => HNil) = HNil
    }

    implicit def hconsPatch[Head, HeadDifference, Tail <: HList, TailDifference <: HList]
    (implicit headPatch: Patch[Head, HeadDifference], tailPatch: Patch[Tail, TailDifference]): Patch[Head :: Tail, HeadDifference :: TailDifference] = {
      new Patch[Head :: Tail, HeadDifference :: TailDifference] {
        override def applyPatch(weight: Head :: Tail, patch: HeadDifference :: TailDifference, learningRate: Double): Head :: Tail = {
          headPatch.applyPatch(weight.head, patch.head, learningRate) :: tailPatch.applyPatch(weight.tail, patch.tail, learningRate)
        }

        override def append(f1: HeadDifference :: TailDifference, f2: => HeadDifference :: TailDifference): HeadDifference :: TailDifference = {
          headPatch.append(f1.head, f2.head) :: tailPatch.append(f1.tail, f2.tail)
        }
      }
    }

    implicit def genericPatch[Data <: Product, Difference <: Product, DataList <: HList, DiffereceList <: HList]
    (
      implicit genericData: Generic.Aux[Data, DataList],
      genericDifference: Generic.Aux[Difference, DiffereceList],
      hlistPatch: Patch[DataList, DiffereceList]
    ) = new Patch[Data, Difference] {
      override def applyPatch(weight: Data, patch: Difference, learningRate: Double): Data = {
        genericData.from(hlistPatch.applyPatch(genericData.to(weight), genericDifference.to(patch), learningRate))
      }

      override def append(f1: Difference, f2: => Difference): Difference = {
        genericDifference.from(hlistPatch.append(genericDifference.to(f1), genericDifference.to(f2)))
      }
    }
  }

  final case class PatchOps[Data, Difference0](override val self: Data, override val patch: Patch[Data, Difference0]) extends Ops[Data] with Differentiable {

    override type Self = Data

    type Difference = Difference0

  }

  trait DifferentiableFunction[-Input, +Output] extends Differentiable {

    type Self >: this.type <: DifferentiableFunction[Input, Output]

    type Difference

    final def self: Self = this

    implicit def patch: Patch[Self, Difference]

    def forward(input: Differentiable.Aux[_ <: Input, _]): DifferentiableFunction.Cache[_ <: Output, input.Difference, Difference]

  }

  object DifferentiableFunction {

    trait Differences[InputDifference, Difference] {
      outer =>

      def inputDifference: InputDifference

      def weightDifference: Difference

    }

    trait Cache[Output, InputDifference, Difference] {

      type OutputDifference

      implicit def outputPatch: Patch[Output, OutputDifference]

      def output: Output

      def backward(difference: OutputDifference): Differences[InputDifference, Difference]

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

    }


    final case class PartialApplied[Input0, Input1, Output, InputDifference0, FDifference, F <: DifferentiableFunction.Aux[Input0, _, F] {
      type Difference = FDifference
    }]
    (input0: Input0, f: F)
    (implicit inputPatch: Patch[Input0, InputDifference0])
      extends DifferentiableFunction[Input1, Output]
        with Cache[PartialApplied[Input0, Input1, Output, InputDifference0, FDifference, F], InputDifference0, FDifference] {

      type Self = PartialApplied[Input0, Input1, Output, InputDifference0, FDifference, F]

      type Difference = PartialApplied.PartialAppliedDifference[InputDifference0, FDifference]

      override implicit def patch: Patch[Self, Difference] = {
        Patch.genericPatch(
          Generic[Self],
          Generic[Difference],
          Patch.hconsPatch(inputPatch, Patch.hconsPatch(f.patch, Patch.HNilPatch))
        )
      }

      override def output: Self = this

      override def forward(input: Differentiable.Aux[_ <: Input1, _]): Cache[_ <: Output, input.Difference, Difference] = ???

      type OutputDifference = Difference

      override implicit def outputPatch: Patch[Self, OutputDifference] = patch

      override def backward(difference: Difference): Difference = difference

    }

    trait PureFunction {
      _: DifferentiableFunction[_, _] with Singleton =>
      override type Self = this.type

      override type Difference = NeverChange.type

      override implicit def patch: Patch[Self, Difference] = Patch.neverChangePatch
    }

    object Multiply extends DifferentiableFunction[INDArray, DifferentiableFunction[INDArray, INDArray]] with PureFunction {
      override def forward(input: Differentiable.Aux[_ <: INDArray, _]): Cache[_ <: PartialApplied[INDArray, INDArray, INDArray, INDArray, NeverChange.type, Multiply.type], input.Difference, Difference] = {
        input match {
          case PatchOps(inputData, Patch.INDArrayPatch) =>
            PartialApplied[INDArray, INDArray, INDArray, INDArray, NeverChange.type, Multiply.type](inputData, Multiply.this)
          case inputPatch =>
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
