package com.thoughtworks


import com.thoughtworks.DeepLearning.Bifunction.MultiplyN.MultipleNPatch
import com.thoughtworks.DeepLearning.Bifunction.Patch.{Delta, NoChange}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms._
import shapeless.{::, DepFn0, DepFn1, DepFn2, HList, HNil, Poly0, PolyApply, the}

import scala.language.existentials
import scala.language.higherKinds
import scalaz.{Apply, Arrow, Category, Choice, Compose, Split, Strong}

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

  sealed trait Bifunction[Input, Output] {

    import Bifunction._

    type Self >: this.type <: Aux[Input, Output, Self]

    def self: Self = this

    type BifunctionPatch <: Patch[Self]

    def forward(input: Input): Cache

    final type Cache = Bifunction.Cache[Output, Patches]
    final type Patches = Bifunction.Patches[Input, BifunctionPatch]

  }

  object Bifunction {

    trait Patch[Weight] {
      def updated(weight: Weight, learningRate: Double): Weight
    }

    object Patch {

      final case class NoChange[Weight]() extends Patch[Weight] {
        override def updated(weight: Weight, learningRate: Double): Weight = weight
      }

      final case class Delta(delta: INDArray) extends Patch[INDArray] {
        override def updated(weight: INDArray, learningRate: Double): INDArray = {
          weight - delta * learningRate
        }
      }

    }

    trait Patches[Input, BifunctionPatch <: Patch[_ <: Ast[_, _]]] {

      def inputPatch: Patch[Input]

      def bifunctionPatch: BifunctionPatch

    }

    trait Cache[Output0, Patches <: Bifunction.Patches[_, _]] {

      type Output <: Output0

      def output: Output

      def backward(outputPatch: Patch[Output]): Patches

    }

    type Aux[Input, Output, F] = Bifunction[Input, Output] {
      type Self = F
    }

    type Ast[Input, Output] = Bifunction[Input, Output]
//    type Ast[Input, Output] = F forSome {type F <: Aux[Input, Output, F]}

    object MultiplyN {

      final case class MultipleNPatch(delta: INDArray) extends Patch[MultiplyN] {
        override def updated(weight: MultiplyN, learningRate: Double) = {
          MultiplyN(weight.n - delta * learningRate)
        }
      }

    }

    final case class MultiplyN(n: INDArray) extends Bifunction[INDArray, INDArray] {
      override type Self = MultiplyN
      override type BifunctionPatch = MultipleNPatch

      override def forward(input: INDArray) = new Cache {

        type Output = INDArray

        override def output = n * input

        override def backward(outputPatch: Patch[INDArray]) = {
          val Delta(deltaOutput) = outputPatch
          new Patches {
            override def bifunctionPatch = MultipleNPatch(deltaOutput * input)

            override def inputPatch = Delta(deltaOutput * n)
          }
        }
      }

    }

    object BinaryMultiply extends Bifunction[INDArray, Ast[INDArray, INDArray]] {
      override type Self = this.type
      override type BifunctionPatch = NoChange[BinaryMultiply.type]

      override def forward(input: INDArray) = new Cache {

        type Output = MultiplyN

        override def output = MultiplyN(input)

        override def backward(outputPatch: Patch[MultiplyN]) = new Patches {

          override def bifunctionPatch: NoChange[BinaryMultiply.type] = NoChange()

          override def inputPatch: Delta = {
            outputPatch match {
              case MultipleNPatch(delta) => Delta(delta)
            }
          }
        }
      }

    }

    //    trait Bifunction2 extends Bifunction[]


    //
    //  object Bifunction {
    //
    //    protected trait BifunctionPatch[F <: Bifunction[_]] extends Patch[F]
    //
    //    type Ast[Input, +Output0] = T forSome {
    //      type T <: Bifunction[Input] {
    //        type Self = T
    //        type Output <: Output0
    //      }
    //    }
    //
    //    sealed trait Patch[Weight] {
    //      def apply(weight: Weight): Weight
    //
    //      def merge(anotherPatch: Patch[Weight]): Patch[Weight]
    //    }
    //
    //    final case class Id[A]() extends Bifunction[A] {
    //
    //      override type Output = A
    //
    //      override type Self = Id[A]
    //
    //      override def forward(input: A) = new Cache {
    //
    //        override def output = input
    //
    //        override def backward(outputPatch: Patch[_ >: A]) = new Patches {
    //
    //          override def bifunctionPatch: BifunctionPatch = NoChange()
    //
    //          override def inputPatch: Patch[_ >: A] = outputPatch
    //        }
    //      }
    //
    //    }
    //
    //    object Compose {
    //
    //      final case class ComposePatch[A, B, C](f: Patch[Ast[B, C]], g: Patch[Ast[A, B]]) extends BifunctionPatch[Compose[A, B, C]] {
    //        override def merge(anotherPatch: Patch[Compose[A, B, C]]): Patch[Compose[A, B, C]] = ???
    //
    //        override def apply(weight: Compose[A, B, C]): Compose[A, B, C] = {
    //          Compose(f.apply(weight.f.self), g.apply(weight.g.self))
    //        }
    //      }
    //
    //    }
    //
    //    final case class Compose[A, B, C](f: Ast[B, C], g: Ast[A, B]) extends Bifunction[A] {
    //      self =>
    //import Compose._
    //      override type Output = C
    //
    //      override type Self = Compose[A, B, C]
    //
    //      override def forward(input: A) = new Cache {
    //
    //        override def output = cacheF.output
    //
    //        lazy val cacheG = g.forward(input)
    //        lazy val cacheF = f.forward(cacheG.output)
    //
    //        override def backward(outputPatch: Patch[_ >: C]) = new Patches {
    //
    //          lazy val patchesF = cacheF.backward(outputPatch)
    //
    //          lazy val patchesG = cacheG.backward(patchesF.inputPatch)
    //
    //          override def bifunctionPatch = ComposePatch(patchesF.bifunctionPatch.asInstanceOf[Patch[Ast[B, C]]], patchesG.bifunctionPatch.asInstanceOf[Patch[Ast[A, B]]])
    //
    //
    //          override def inputPatch: Patch[_ >: A] = patchesG.inputPatch
    //
    //        }
    //
    //      }
    //
    //    }
    //
    //    object Bifunction2 {
    //
    //      final case class PartialAppliedBifunction[Input0, Input1, Output0, F <: Bifunction2[Input0, Input1]]
    //      (input0: Input0, f: F {type Output2 = Output0})
    //        extends Bifunction[Input1] {
    //
    //        type Output = Output0
    //
    //        type Self = PartialAppliedBifunction[Input0, Input1, Output0, F]
    //
    //        override def forward(input1: Input1) = new Cache {
    //          val cache2 = f.forward2(input0, input1)
    //
    //          override def output = cache2.output
    //
    //          override def backward(outputPatch: Patch[_ >: Output]) = new Patches {
    //            val patches = cache2.backward(outputPatch)
    //
    //            override def bifunctionPatch = PartialAppliedPatch[Input0, Input1, Output0, F](patches.input0Patch)
    //
    //            override def inputPatch: Patch[_ >: Input1] = patches.input1Patch
    //          }
    //        }
    //
    //      }
    //
    //      final case class PartialAppliedPatch[Input0, Input1, Output0, F <: Bifunction2[Input0, Input1]]
    //      (input0Patch: Patch[_ >: Input0])
    //        extends BifunctionPatch[PartialAppliedBifunction[Input0, Input1, Output0, F]] {
    //
    //        override def apply(weight: PartialAppliedBifunction[Input0, Input1, Output0, F]): PartialAppliedBifunction[Input0, Input1, Output0, F] = {
    //          PartialAppliedBifunction(input0Patch.apply(weight.input0).asInstanceOf[Input0], weight.f)
    //        }
    //
    //        override def merge(anotherPatch: Patch[PartialAppliedBifunction[Input0, Input1, Output0, F]]): Patch[PartialAppliedBifunction[Input0, Input1, Output0, F]] = {
    //          anotherPatch match {
    //            case PartialAppliedPatch(anotherPatches) =>
    //              PartialAppliedPatch(input0Patch.asInstanceOf[Patch[Input0]].merge(anotherPatches.asInstanceOf[Patch[Input0]]))
    //          }
    //        }
    //      }
    //
    //    }
    //
    //    trait Bifunction2[Input0, Input1] extends Bifunction[Input0] {
    //
    //      import Bifunction2._
    //
    //      type Output2
    //
    //      type Output = PartialAppliedBifunction[Input0, Input1, Output2, Self]
    //
    //      type Self >: this.type <: Bifunction2[Input0, Input1]
    //
    //      trait Patches2 {
    //        def input0Patch: Patch[_ >: Input0]
    //
    //        def input1Patch: Patch[_ >: Input1]
    //      }
    //
    //      trait Cache2 {
    //        def output: Output2
    //
    //        def backward(outputPatch: Patch[_ >: Output2]): Patches2
    //      }
    //
    //      def forward2(input0: Input0, input1: Input1): Cache2
    //
    //      override def forward(input0: Input0) = new Cache {
    //
    //        override val output: Output = new Output(input0, Bifunction2.this)
    //
    //        override def backward(outputPatch: Patch[_ >: Output]) = new Patches {
    //          override def bifunctionPatch = NoChange()
    //
    //          override def inputPatch: Patch[_ >: Input0] = outputPatch match {
    //            case PartialAppliedPatch(input0Patch) => input0Patch.asInstanceOf[Patch[_ >: Input0]]
    //          }
    //        }
    //
    //      }
    //
    //    }
    //
    //    final case class NoChange[Weight <: Bifunction[_]]() extends BifunctionPatch[Weight] {
    //      override def merge(anotherPatch: Patch[Weight]): Patch[Weight] = anotherPatch
    //
    //      override def apply(weight: Weight): Weight = weight
    //    }
    //
    //    final case object Multiply extends Bifunction2[INDArray, INDArray] {
    //      override type Output2 = INDArray
    //      override type Self = Multiply.type
    //
    //      override def forward2(input0: INDArray, input1: INDArray): Cache2 = ???
    //
    //    }
    //
    //    final case class Substitute[A, B, C](x: Ast[A, Ast[B, C]], y: Ast[A, B]) extends Bifunction[A] {
    //
    //      override type Output = C
    //
    //      override def forward(input: A) = new Cache {
    //
    //        lazy val cacheX0 = x.forward(input)
    //        lazy val cacheY = y.forward(input)
    //        lazy val x1: Ast[B, C] = cacheX0.output
    //        lazy val cacheX1 = x1.forward(cacheY.output)
    //
    //        override def output: C = cacheX1.output
    //
    //        override def backward(outputPatch: Patch[_ >: C]) = new Patches {
    //
    //          lazy val patchesX1 = cacheX1.backward(outputPatch)
    //
    //          lazy val patchesY = cacheY.backward(patchesX1.inputPatch)
    //
    //          lazy val patchesX = cacheX0.backward(patchesX1.bifunctionPatch.asInstanceOf[Patch[Ast[B, C]]])
    //
    //          override def bifunctionPatch = new BifunctionPatch {
    //            override def apply(weight: Self): Self = {
    //              Substitute[A, B, C](patchesX.bifunctionPatch.asInstanceOf[Patch[Ast[A, Ast[B, C]]]].apply(weight.x), patchesY.bifunctionPatch.asInstanceOf[Patch[Ast[A, B]]].apply(weight.y))
    //            }
    //
    //            override def merge(anotherPatch: Patch[Self]) = ???
    //          }
    //
    //          override def inputPatch: Patch[A] = patchesX.inputPatch.asInstanceOf[Patch[A]].merge(patchesY.inputPatch.asInstanceOf[Patch[A]])
    //
    //        }
    //      }
    //
    //      override type Self = Substitute[A, B, C]
    //    }
    //
    implicit object BifunctionInstances extends SKICombinator[Ast] with Multiply[Ast] {
      //      override def id[A] = Id[A]
      //
      //      override def compose[A, B, C](f: Ast[B, C], g: Ast[A, B]) = new Compose[A, B, C](f, g)
      //
      //      def constant[A, B, C](x: Ast[A, B]): Ast[C, Ast[A, B]] = ???
      //
      //      def substitute[A, B, C](x: Ast[A, Ast[B, C]], y: Ast[A, B]): Ast[A, C] = Substitute[A, B, C](x, y)
      //
      override def multiply = BinaryMultiply

      //
      //
      override def compose[A, B, C](f: Ast[B, C], g: Ast[A, B]): Ast[A, C] = ???

      override def constant[A, B, C](x: Ast[A, B]): Ast[C, Ast[A, B]] = ???

      override def id[A]: Ast[A, A] = ???

      override def substitute[A, B, C](x: Ast[A, Ast[B, C]], y: Ast[A, B]): Ast[A, C] = ???
    }

    //
  }

  //

}
