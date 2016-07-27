package com.thoughtworks


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms._
import shapeless.{::, DepFn0, DepFn1, DepFn2, HList, HNil, Poly0, PolyApply, the}

import scala.language.higherKinds
import scalaz.{Apply, Arrow, Category, Choice, Compose, Split, Strong}

object DeepLearning {


  sealed trait Bifunction[-Input, +Output] {

    import Bifunction._

    type Repr >: this.type <: Bifunction[Input, Output]

    trait Patches {

      def thisPatch: BifunctionPatch

      def inputPatch: Patch[_ >: Input]

    }

    trait BifunctionPatch extends Patch[Repr]

    trait Cache {
      def output: Output

      def backward(outputPatch: Patch[_ >: Output]): Patches
    }

    def forward(input: Input): Cache

    def self: Repr = this

  }

  object Bifunction {

    sealed trait Patch[Weight] {
      def apply(weight: Weight): Weight

      def merge(anotherPatch: Patch[Weight]): Patch[Weight]
    }

    final case class Id[A]() extends Bifunction[A, A] {

      override type Repr = Id[A]

      override def forward(input: A) = new Cache {

        override def output = input

        override def backward(outputPatch: Patch[_ >: A]) = new Patches {

          override def thisPatch: BifunctionPatch = new BifunctionPatch {

            override def merge(anotherPatch: Patch[Id[A]]): Patch[Id[A]] = this

            override def apply(weight: Id[A]): Id[A] = weight

          }

          override def inputPatch: Patch[_ >: A] = outputPatch
        }
      }

    }

    type HasRepr[B, C] = F forSome {
      type F <: Bifunction[B, C] {type Repr = F}
    }

    final case class Compose[A, B, C, F <: Bifunction[B, C] {type Repr = F}, G <: Bifunction[A, B] {type Repr = G}](f: F, g: G) extends Bifunction[A, C] {
      self =>

      override type Repr = Compose[A, B, C, F, G]

      override def forward(input: A) = new Cache {

        override def output = cacheF.output

        lazy val cacheG = g.forward(input)
        lazy val cacheF = f.forward(cacheG.output)

        override def backward(outputPatch: Patch[_ >: C]) = new Patches {

          lazy val patchesF = cacheF.backward(outputPatch)

          lazy val patchesG = cacheG.backward(patchesF.inputPatch)

          override def thisPatch = new BifunctionPatch {

            override def apply(weight: Compose[A, B, C, F, G]): Compose[A, B, C, F, G] = {
              Compose(patchesF.thisPatch.apply(weight.f), patchesG.thisPatch.apply(weight.g))
            }

            override def merge(anotherPatch: Patch[Compose[A, B, C, F, G]]): Patch[Compose[A, B, C, F, G]] = ???

          }

          override def inputPatch: Patch[_ >: A] = patchesG.inputPatch

        }

      }

    }

    object BifunctionCategory extends Category[HasRepr] {
      override def id[A]: HasRepr[A, A] = Id[A]

      override def compose[A, B, C](f: HasRepr[B, C], g: HasRepr[A, B]): HasRepr[A, C] = new Compose[A, B, C, f.Repr, g.Repr](f.self, g.self)
    }

  }


}
