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

    implicit def wrapperPatch[Wrapper, Underlying, Difference](implicit genereic: Generic.Aux[Wrapper, Underlying :: HNil], underlyingPatch: Patch[Underlying, Difference]) = new Patch[Wrapper, Difference] {
      override def applyPatch(weight: Wrapper, patch: Difference, learningRate: Double): Wrapper = {
        genereic.from(underlyingPatch.applyPatch(genereic.to(weight).head, patch, learningRate) :: HNil)
      }

      override def append(f1: Difference, f2: => Difference): Difference = underlyingPatch.append(f1, f2)
    }

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


    trait PartialApplied[InputDifference0, FDifference] {
      _: DifferentiableFunction[_, _] with Cache[_, InputDifference0, FDifference] =>

      type Difference = PartialApplied.PartialAppliedDifference[InputDifference0, FDifference]

      override def output: Self = this

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


    final case class PartialAppliedMultiply
    (input0Data: INDArray, outer: Multiply.type)
    (implicit protected val inputPatch: Patch[INDArray, INDArray])
      extends DifferentiableFunction[INDArray, INDArray]
        with Cache[PartialAppliedMultiply, INDArray, NeverChange.type]
        with PartialApplied[INDArray, NeverChange.type] {

      type Self = PartialAppliedMultiply

      override implicit def patch: Patch[Self, Difference] = {
        Patch.genericPatch(
          Generic[Self],
          Generic[Difference],
          Patch.hconsPatch(inputPatch, Patch.hconsPatch(outer.patch, Patch.HNilPatch))
        )
      }

      override def forward(input1: Differentiable.Aux[_ <: INDArray, _]): Cache[INDArray, input1.Difference, Difference] = {
        input1 match {
          case PatchOps(input1Data, Patch.INDArrayPatch) =>
            new Cache[INDArray, INDArray, Difference] {
              type OutputDifference = INDArray

              override implicit def outputPatch: Patch[INDArray, OutputDifference] = Patch.INDArrayPatch

              override def output: INDArray = input0Data * input1Data

              override def backward(difference: OutputDifference) = new Differences[INDArray, Difference] {
                override def inputDifference: INDArray = input0Data

                override def weightDifference: Difference = new Difference(input1Data, NeverChange)
              }
            }
          case _ =>
            throw new IllegalArgumentException(s"Unsupported patch type ${input1}")

        }
      }.unsafeCast
    }

    object Multiply extends DifferentiableFunction[INDArray, DifferentiableFunction[INDArray, INDArray]] with PureFunction {

      override def forward(input0: Differentiable.Aux[_ <: INDArray, _]): Cache[_ <: PartialAppliedMultiply, input0.Difference, Difference] = {
        input0 match {
          case PatchOps(inputData, Patch.INDArrayPatch) =>
            PartialAppliedMultiply(inputData, Multiply.this)
          case _ =>
            throw new IllegalArgumentException(s"Unsupported patch type ${input0}")
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

