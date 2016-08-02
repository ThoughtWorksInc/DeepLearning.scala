package com.thoughtworks


import com.thoughtworks.DeepLearning.Differentiable.Aux
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

    type Self >: this.type <: DifferentiableFunction.Aux[Input, Output, Self]

    type Difference

    final def self: Self = this

    implicit def patch: Patch[Self, Difference]

    def forward[InputData <: Input, InputDifference](input: Differentiable.Aux[InputData, InputDifference]): DifferentiableFunction.Cache[_ <: Output, InputDifference, Difference]

  }

  object DifferentiableFunction {

    trait Differences[InputDifference, Difference] {
      outer =>

      def inputDifference: InputDifference

      def weightDifference: Difference

    }

    trait Cache[Output0, InputDifference, Difference] {

      type Output = Output0

      type OutputDifference

      def output: Differentiable.Aux[Output, OutputDifference]

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

      override def forward[InputData <: INDArray, InputDifference](input1: Differentiable.Aux[InputData, InputDifference]): Cache[INDArray, InputDifference, Difference] = {
        input1 match {
          case PatchOps(input1Data, Patch.INDArrayPatch) =>
            new Cache[INDArray, INDArray, Difference] {
              type OutputDifference = INDArray

              override def output = PatchOps(input0Data * input1Data, Patch.INDArrayPatch)

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

    object Multiply extends DifferentiableFunction[INDArray, PartialAppliedMultiply] with PureFunction {

      override def forward[InputData <: INDArray, InputDifference](input0: Differentiable.Aux[InputData, InputDifference]): Cache[PartialAppliedMultiply, InputDifference, Difference] = {
        input0 match {
          case PatchOps(inputData, Patch.INDArrayPatch) =>
            PartialAppliedMultiply(inputData, Multiply.this)
          case _ =>
            throw new IllegalArgumentException(s"Unsupported patch type ${input0}")
        }
      }.unsafeCast
    }

    final case class Compose[A, B, C, F <: DifferentiableFunction.Aux[B, C, F], G <: DifferentiableFunction.Aux[A, B, G]](f: F, g: G) extends DifferentiableFunction[A, C] {
      outer =>

      override type Self = Compose[A, B, C, F, G]

      override type Difference = (f.Difference, g.Difference)

      override def forward[InputData <: A, InputDifference](input: Differentiable.Aux[InputData, InputDifference]): Cache[_ <: C, InputDifference, Difference] = {
        val cacheG: Cache[_ <: B, InputDifference, g.Difference] = g.forward(input)
        val cacheF: Cache[_ <: C, cacheG.OutputDifference, f.Difference] = f.forward[cacheG.Output, cacheG.OutputDifference](cacheG.output).unsafeCast
        new Cache[cacheF.Output, InputDifference, Difference] {

          override type OutputDifference = cacheF.OutputDifference

          override def backward(difference: OutputDifference): Differences[input.Difference, (f.Difference, g.Difference)] = {

            val differencesF: Differences[cacheG.OutputDifference, f.Difference] = cacheF.backward(difference)

            val differencesG = cacheG.backward(differencesF.inputDifference)

            new Differences[InputDifference, (f.Difference, g.Difference)] {
              override def inputDifference: InputDifference = differencesG.inputDifference

              override def weightDifference: (f.Difference, g.Difference) = (differencesF.weightDifference, differencesG.weightDifference)
            }

          }

          override def output: Differentiable.Aux[Output, cacheF.OutputDifference] = cacheF.output

        }

      }

      override implicit def patch: Patch[Self, Difference] = {
        Patch.genericPatch(Generic[Self], Generic[Difference], Patch.hconsPatch(f.patch, Patch.hconsPatch(g.patch, Patch.HNilPatch)))
      }
    }

    implicit object DifferentiableFunctionInstances extends Split[DifferentiableFunction] with Category[DifferentiableFunction] with Multiply[DifferentiableFunction] {

      override def multiply: DifferentiableFunction[INDArray, DifferentiableFunction[INDArray, INDArray]] = Multiply

      override def compose[A, B, C](f: DifferentiableFunction[B, C], g: DifferentiableFunction[A, B]) = {
        new Compose[A, B, C, f.Self, g.Self](f, g)
      }

      override def split[A, B, C, D](f: DifferentiableFunction[A, B], g: DifferentiableFunction[C, D]): DifferentiableFunction[(A, C), (B, D)] = {
        ???
      }

      override def id[A]: DifferentiableFunction[A, A] = ???

    }

  }

}

