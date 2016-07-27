package com.thoughtworks


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms._
import shapeless.{::, DepFn0, DepFn1, DepFn2, HList, HNil, Poly0, PolyApply, the}

import scala.language.existentials
import scala.language.higherKinds
import scalaz.{Apply, Arrow, Category, Choice, Compose, Split, Strong}

object DeepLearning {

  trait Multiply[=>:[_, _]] {
    def multiply: INDArray =>: INDArray =>: INDArray
  }

  trait HasRepr {

    type Repr >: this.type <: HasRepr

    def self: Repr = this

  }

  sealed trait Bifunction[-Input, +Output] extends HasRepr {

    import Bifunction._

    trait Patches {

      def functionPatch: BifunctionPatch

      def inputPatch: Patch[_ >: Input]

    }

    trait BifunctionPatch extends Patch[Repr]

    trait Cache {
      def output: Output

      def backward(outputPatch: Patch[_ >: Output]): Patches
    }

    def forward(input: Input): Cache

  }

  object Bifunction {

    type Ast[-Input, +Output] = Self forSome {
      type Self <: Bifunction[Input, Output] {type Repr = Self}
    }

    sealed trait Patch[Weight] {
      def apply(weight: Weight): Weight

      def merge(anotherPatch: Patch[Weight]): Patch[Weight]
    }

    final case class Id[A]() extends Bifunction[A, A] {

      override type Repr = Id[A]

      override def forward(input: A) = new Cache {

        override def output = input

        override def backward(outputPatch: Patch[_ >: A]) = new Patches {

          override def functionPatch: BifunctionPatch = new BifunctionPatch {

            override def merge(anotherPatch: Patch[Id[A]]): Patch[Id[A]] = this

            override def apply(weight: Id[A]): Id[A] = weight

          }

          override def inputPatch: Patch[_ >: A] = outputPatch
        }
      }

    }


    final case class Compose[A, B, C](f: Ast[B, C], g: Ast[A, B]) extends Bifunction[A, C] {
      self =>

      override type Repr = Compose[A, B, C]

      override def forward(input: A) = new Cache {

        override def output = cacheF.output

        lazy val cacheG = g.forward(input)
        lazy val cacheF = f.forward(cacheG.output)

        override def backward(outputPatch: Patch[_ >: C]) = new Patches {

          lazy val patchesF = cacheF.backward(outputPatch)

          lazy val patchesG = cacheG.backward(patchesF.inputPatch)

          override def functionPatch = new BifunctionPatch {

            override def apply(weight: Compose[A, B, C]): Compose[A, B, C] = {
              Compose(patchesF.functionPatch.apply(weight.f.self), patchesG.functionPatch.apply(weight.g.self))
            }

            override def merge(anotherPatch: Patch[Compose[A, B, C]]): Patch[Compose[A, B, C]] = ???

          }

          override def inputPatch: Patch[_ >: A] = patchesG.inputPatch

        }

      }

    }

    final case class PartialAppliedBifunction[Input0, Input1, Output, F <: Bifunction2[Input0, Input1, Output]](input0: Input0, f: F) extends Bifunction[Input1, Output] {


      type Repr = PartialAppliedBifunction[Input0, Input1, Output, F]

      final case class CurriedPatch(input0Patch: Patch[_ >: Input0]) extends BifunctionPatch {

        override def apply(weight: Repr): Repr = {
          PartialAppliedBifunction(input0Patch.asInstanceOf[Patch[Input0]].apply(weight.input0), f)
        }

        override def merge(anotherPatch: Patch[Repr]): Patch[Repr] = {
          anotherPatch match {
            case CurriedPatch(anotherPatches) =>
              CurriedPatch(input0Patch.asInstanceOf[Patch[Input0]].merge(anotherPatches.asInstanceOf[Patch[Input0]]))
          }
        }

      }

      override def forward(input1: Input1) = new Cache {
        val cache2 = f.forward2(input0, input1)

        override def output: Output = cache2.output

        override def backward(outputPatch: Patch[_ >: Output]) = new Patches {
          val patches = cache2.backward(outputPatch)

          override def functionPatch = CurriedPatch(patches.input0Patch)

          override def inputPatch: Patch[_ >: Input1] = patches.input1Patch
        }
      }

    }

    trait Bifunction2[Input0, Input1, Output] extends Bifunction[Input0, PartialAppliedBifunction[Input0, Input1, Output, _]] {

      type Repr >: this.type <: Bifunction2[Input0, Input1, Output]

      trait Patches2 {
        def input0Patch: Patch[_ >: Input0]

        def input1Patch: Patch[_ >: Input1]
      }

      trait Cache2 {
        def output: Output

        def backward(outputPatch: Patch[_ >: Output]): Patches2
      }

      def forward2(input0: Input0, input1: Input1): Cache2

      override def forward(input0: Input0) = new Cache {

        override val output = PartialAppliedBifunction[Input0, Input1, Output, Repr](input0, Bifunction2.this)

        override def backward(outputPatch: Patch[_ >: PartialAppliedBifunction[Input0, Input1, Output, _]]) = new Bifunction2.this.Patches {
          override def functionPatch = new Bifunction2.this.BifunctionPatch {

            override def apply(weight: Bifunction2.this.Repr): Bifunction2.this.Repr = weight

            override def merge(anotherPatch: Patch[Bifunction2.this.Repr]): Patch[Bifunction2.this.Repr] = anotherPatch
          }

          override def inputPatch: Patch[_ >: Input0] = outputPatch match {
            case output.CurriedPatch(input0Patch) => input0Patch
          }
        }

      }

    }

    final case object Multiply extends Bifunction2[INDArray, INDArray, INDArray] {
      type Repr = Multiply.type

      override def forward2(input0: INDArray, input1: INDArray): _root_.com.thoughtworks.DeepLearning.Bifunction.Multiply.Cache2 = ???

    }

    implicit object BifunctionInstances extends Category[Ast] with Multiply[Ast] {
      override def id[A] = Id[A]

      override def compose[A, B, C](f: Ast[B, C], g: Ast[A, B]) = new Compose[A, B, C](f, g)

      override def multiply = Multiply
    }

  }


}
