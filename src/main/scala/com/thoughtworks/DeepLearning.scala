package com.thoughtworks


import com.thoughtworks.DeepLearning.Differentiable.Aux
import com.thoughtworks.DeepLearning.NeverChange
import com.thoughtworks.DeepLearning.Patch.PairPatch
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms._
import shapeless.{::, DepFn0, DepFn1, DepFn2, Generic, HList, HNil, Poly0, PolyApply, Widen, the}

import scala.language.existentials
import scala.language.higherKinds
import scalaz.syntax.Ops
import scalaz.{Apply, Arrow, Category, Choice, Compose, Monoid, Semigroup, Split, Strong}

object DeepLearning {

  trait Multiply[=>:[_, _]] {
    def multiply: INDArray =>: INDArray =>: INDArray
  }

  sealed trait Patch[Data, Difference] extends Monoid[Difference] {

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

    final case class PairPatch[Data0, Data1, Difference0, Difference1](implicit patch0: Patch[Data0, Difference0], patch1: Patch[Data1, Difference1]) extends Patch[(Data0, Data1), (Difference0, Difference1)] {
      override def applyPatch(weight: (Data0, Data1), patch: (Difference0, Difference1), learningRate: Double): (Data0, Data1) = {
        (patch0.applyPatch(weight._1, patch._1, learningRate), patch1.applyPatch(weight._2, patch._2, learningRate))
      }

      override def zero: (Difference0, Difference1) = {
        (patch0.zero, patch1.zero)
      }

      override def append(f1: (Difference0, Difference1), f2: => (Difference0, Difference1)): (Difference0, Difference1) = {
        (patch0.append(f1._1, f2._1), patch1.append(f1._2, f2._2))
      }
    }

    implicit def wrapperPatch[Wrapper, Underlying, Difference](implicit genereic: Generic.Aux[Wrapper, Underlying :: HNil], underlyingPatch: Patch[Underlying, Difference]) = new Patch[Wrapper, Difference] {
      override def applyPatch(weight: Wrapper, patch: Difference, learningRate: Double): Wrapper = {
        genereic.from(underlyingPatch.applyPatch(genereic.to(weight).head, patch, learningRate) :: HNil)
      }

      override def append(f1: Difference, f2: => Difference): Difference = underlyingPatch.append(f1, f2)

      override def zero: Difference = underlyingPatch.zero
    }

    implicit object INDArrayPatch extends Patch[INDArray, Option[INDArray]] {
      override def applyPatch(weight: INDArray, patch: Option[INDArray], learningRate: Double): INDArray = {
        patch match {
          case None =>
            weight
          case Some(delta) =>
            weight + delta * learningRate
        }
      }

      override def append(f1: Option[INDArray], f2: => Option[INDArray]): Option[INDArray] = {
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

      override def zero = None
    }

    final case class NeverChangePatch[Data, Difference >: NeverChange.type]() extends Patch[Data, Difference] {
      override def applyPatch(weight: Data, patch: Difference, learningRate: Double) = weight

      override def append(f1: Difference, f2: => Difference) = NeverChange

      override def zero = NeverChange
    }

    implicit def neverChangePatch[Data <: Singleton] = new NeverChangePatch[Data, NeverChange.type]

    implicit object HNilPatch extends Patch[HNil, HNil] {
      override def applyPatch(weight: HNil, patch: HNil, learningRate: Double) = HNil

      override def append(f1: HNil, f2: => HNil) = HNil

      override def zero = HNil
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

        override def zero: HeadDifference :: TailDifference = headPatch.zero :: tailPatch.zero
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

      override def zero: Difference = {
        genericDifference.from(hlistPatch.zero)
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

    trait Differences[+InputDifference, +Difference] {

      def inputDifference: InputDifference

      def weightDifference: Difference

    }

    trait Cache[Output0, +InputDifference, +Difference] {

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
      _: DifferentiableFunction[_, _] =>
      override type Self = this.type

      override type Difference = NeverChange.type

      override implicit def patch = Patch.NeverChangePatch[Self, Difference]()
    }


    final case class PartialAppliedMultiply
    (input0Data: INDArray, outer: Multiply.type)
    (implicit protected val inputPatch: Patch[INDArray, Option[INDArray]])
      extends DifferentiableFunction[INDArray, INDArray]
        with Cache[PartialAppliedMultiply, Option[INDArray], NeverChange.type]
        with PartialApplied[Option[INDArray], NeverChange.type] {

      type Self = PartialAppliedMultiply

      override implicit def patch: Patch[Self, Difference] = {
        Patch.genericPatch(
          Generic[Self],
          Generic[Difference],
          Patch.hconsPatch(inputPatch, Patch.hconsPatch(outer.patch, Patch.HNilPatch))
        )
      }

      override def forward[InputData <: INDArray, InputDifference](input1: Differentiable.Aux[InputData, InputDifference]): Cache[INDArray, InputDifference, Difference] = {
        type ExpectedDifferentiable = Differentiable.Aux[_ <: INDArray, _ >: Option[INDArray]]
        input1 match {
          case differentiable1: ExpectedDifferentiable =>
            new Cache[INDArray, INDArray, Difference] {
              type OutputDifference = Option[INDArray]

              override def output = PatchOps(input0Data * differentiable1.self, Patch.INDArrayPatch)

              override def backward(difference: OutputDifference) = new Differences[INDArray, Difference] {
                override def inputDifference: INDArray = input0Data

                override def weightDifference: Difference = new Difference(Some(differentiable1.self), NeverChange)
              }
            }
        }
      }.unsafeCast
    }

    object Multiply extends DifferentiableFunction[INDArray, PartialAppliedMultiply] with PureFunction {

      override def forward[InputData <: INDArray, InputDifference](input0: Differentiable.Aux[InputData, InputDifference]): Cache[PartialAppliedMultiply, InputDifference, Difference] = {
        type ExpectedDifferentiable = Differentiable.Aux[_ <: INDArray, _ >: Option[INDArray]]
        input0 match {
          case differentiable0: ExpectedDifferentiable =>
            PartialAppliedMultiply(differentiable0.self, Multiply.this)
        }
      }.unsafeCast
    }

    final case class Compose[A, B, C, F <: DifferentiableFunction.Aux[B, C, F], G <: DifferentiableFunction.Aux[A, B, G]](f: F, g: G) extends DifferentiableFunction[A, C] {

      override type Self = Compose[A, B, C, F, G]

      override type Difference = (f.Difference, g.Difference)

      override def forward[InputData <: A, InputDifference](input: Differentiable.Aux[InputData, InputDifference]): Cache[_ <: C, InputDifference, Difference] = {
        val cacheG: Cache[_ <: B, InputDifference, g.Difference] = g.forward(input)
        val cacheF: Cache[_ <: C, cacheG.OutputDifference, f.Difference] = f.forward[cacheG.Output, cacheG.OutputDifference](cacheG.output)
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

    final case class Id[A]() extends DifferentiableFunction[A, A] {
      override type Self = Id[A]
      override type Difference = NeverChange.type

      override implicit def patch = Patch.NeverChangePatch[Self, Difference]()

      override def forward[InputData <: A, InputDifference](input: Differentiable.Aux[InputData, InputDifference]) = {
        new Cache[InputData, InputDifference, NeverChange.type] {
          override type OutputDifference = InputDifference

          override def output = input

          override def backward(difference: OutputDifference) = new Differences[InputDifference, NeverChange.type] {
            override def inputDifference = difference

            override def weightDifference = NeverChange
          }
        }
      }
    }

    final case class Arr[A, B](f: A => B) extends DifferentiableFunction[A, B] {
      override type Self = Arr[A, B]

      override type Difference = NeverChange.type

      override implicit def patch = Patch.NeverChangePatch[Self, Difference]()

      override def forward[InputData <: A, InputDifference](input: Differentiable.Aux[InputData, InputDifference]) = new Cache[B, InputDifference, Difference] {
        override type OutputDifference = Any

        override def output: Differentiable.Aux[Output, Any] = {
          PatchOps(f(input.self), new Patch[Output, Any] {
            override def applyPatch(weight: Output, patch: Any, learningRate: Double): Output = weight

            override def zero: Any = NeverChange

            override def append(f1: Any, f2: => Any) = NeverChange
          })
        }

        override def backward(difference: Any) = new Differences[InputDifference, NeverChange.type] {
          override def inputDifference: InputDifference = input.patch.zero

          override def weightDifference = NeverChange
        }
      }
    }

    final case class First[A, B, C, FA <: DifferentiableFunction.Aux[A, B, FA]](fa: FA) extends DifferentiableFunction[(A, C), (B, C)] {

      type Self = First[A, B, C, FA]

      override type Difference = fa.Difference

      override implicit def patch = {
        Patch.wrapperPatch[Self, FA, Difference](Generic[Self], fa.patch)
      }

      override def forward[InputData <: (A, C), InputDifference](input: Differentiable.Aux[InputData, InputDifference]) = {
        val (patch0: Patch[A, _], patch1: Patch[C, _]) = input.patch match {
          case pairPatch: Patch.PairPatch[A, C, _, _] =>
            (pairPatch.patch0, pairPatch.patch1)
          case Patch.NeverChangePatch() =>
            (Patch.NeverChangePatch(), Patch.NeverChangePatch())
          case _ =>
            throw new IllegalArgumentException
        }
        val (a, c) = input.self
        val differentiableA = PatchOps(a, patch0)
        val differentiableC = PatchOps(c, patch1)
        val forwardFa = fa.forward(differentiableA)

        val differentiableB = forwardFa.output
        new Cache[(differentiableB.Self, differentiableC.Self), InputDifference, Difference] {
          override type OutputDifference = (forwardFa.OutputDifference, differentiableC.Difference)

          override def output = {
            PatchOps[(differentiableB.Self, differentiableC.Self), OutputDifference](
              (differentiableB.self: differentiableB.Self) -> (c: differentiableC.Self),
              new PairPatch()(differentiableB.patch, patch1.asInstanceOf[Patch[differentiableC.Self, differentiableC.Difference]])
            )
          }

          override def backward(difference: OutputDifference) = {
            val differencesB = forwardFa.backward(difference._1)
            new Differences[InputDifference, Difference] {
              override def inputDifference: InputDifference = {
                (differencesB.inputDifference, difference._2).asInstanceOf[InputDifference]
              }
              override def weightDifference: fa.Difference = {
                differencesB.weightDifference
              }
            }
          }
        }
      }.unsafeCast
    }

    implicit object DifferentiableFunctionInstances extends Arrow[DifferentiableFunction] with Multiply[DifferentiableFunction] {

      override def multiply = Multiply

      override def compose[A, B, C](f: DifferentiableFunction[B, C], g: DifferentiableFunction[A, B]) = {
        new Compose[A, B, C, f.Self, g.Self](f, g)
      }

      override def id[A] = Id[A]

      override def arr[A, B](f: (A) => B) = Arr(f)

      override def first[A, B, C](fa: DifferentiableFunction[A, B]) = First[A, B, C, fa.Self](fa)

    }

  }

}

