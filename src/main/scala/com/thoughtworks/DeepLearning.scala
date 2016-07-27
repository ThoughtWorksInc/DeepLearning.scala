package com.thoughtworks


import com.thoughtworks.DeepLearning.Bifunction.Add
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms._
import shapeless.{::, DepFn0, DepFn1, DepFn2, HList, HNil, Poly0, PolyApply, the}

import scala.language.higherKinds
import scalaz.{Apply, Arrow, Category, Choice, Compose, Split, Strong}

object DeepLearning {


  sealed trait Patch[Weight] extends Any {
    //    def apply(weight: Weight, learningRate: Double): Weight

    def merge(anotherPatch: Patch[Weight]): Patch[Weight]
  }

  final case class CurriedPatch[A, B](patchA: Patch[A], patchB: Patch[B]) extends Patch[Bifunction[A, B]] {
    override def merge(anotherPatch: Patch[Bifunction[A, B]]): Patch[Bifunction[A, B]] = {
      anotherPatch match {
        case CurriedPatch(anotherPatchA, anotherPatchB) => CurriedPatch(patchA.merge(anotherPatchA), patchB.merge(anotherPatchB))
      }
    }
  }

  final case class Delta(delta: INDArray) extends AnyVal with Patch[INDArray] {
    //    override def apply(weight: INDArray, learningRate: Double): INDArray = {
    //      weight - delta * learningRate
    //    }

    override def merge(anotherPatch: Patch[INDArray]): Patch[INDArray] = {
      anotherPatch match {
        case Delta(anotherDelta) => Delta(delta + anotherDelta)
      }
    }
  }

  trait ForwardPass[Cache, Output] {
    def output: Output

    def cache: Cache
  }

  sealed trait Bifunction[Input, Output] {

    type Cache

    def forward(input: Input): ForwardPass[Cache, Output]

    def backward(cache: Cache, patchOutput: Patch[Output]): Patch[Input]

  }

  trait Exp[=>:[_, _]] {
    def exp: INDArray =>: INDArray
  }

  trait Add[=>:[_, _]] {
    def add: INDArray =>: INDArray =>: INDArray
  }

  trait Substitution[=>:[_, _]] {
    def substitute[A, B, C](x: A =>: B =>: C, y: A =>: B): A =>: C
  }

  trait Constant[=>:[_, _]] {
    def constant[A, B, C](x: A =>: B): C =>: A =>: B
  }

  trait SKICombinator[=>:[_, _]] extends Substitution[=>:] with Constant[=>:] with Category[=>:]

  object Bifunction {

    final case class Id[A]() extends Bifunction[A, A] {
      override type Cache = Unit

      override def forward(input: A) = new ForwardPass[Unit, A] {

        override def output: A = input

        override def cache: Unit = ()
      }

      override def backward(cache: Unit, patchOutput: Patch[A]): Patch[A] = patchOutput
    }

    final case class PartiallyAppliedBifunction[Input0, Input1, Output](input0: Input0, bifunction2: Bifunction2[Input0, Input1, Output]) extends Bifunction[Input1, Output] {

      type Cache = bifunction2.Cache2

      override def forward(input1: Input1): ForwardPass[Cache, Output] = {
        bifunction2.forward2(input0, input1)
      }

      override def backward(cache: Cache, patchOutput: Patch[Output]): Patch[Input1] = {
        bifunction2.backward2(cache, patchOutput).patch1
      }

    }

    trait Patch2[Input0, Input1] {
      def patch0: Patch[Input0]
      def patch1: Patch[Input1]
    }

    trait Bifunction2[Input0, Input1, Output] extends Bifunction[Input0, Bifunction[Input1, Output]] {

      override def forward(input: Input0): Bifunction[Input1, Output] = PartiallyAppliedBifunction[Input0, Input1, Output](input, this)

      override def backward(cache: Cache, patchOutput: Patch[Bifunction[Input1, Output]]): Patch[Input0] = {

        ???
      }

      type Cache = Unit

      type Cache2

      def forward2(input0: Input0, input1: Input1): ForwardPass[Cache2, Output]
      def backward2(cache2: Cache2, patchOutput: Patch[Output]): Patch2[Input0, Input1]

    }

    object Add extends Bifunction2[INDArray, INDArray, INDArray] {
      override type Cache2 = (INDArray, INDArray)

      override def forward2(input0: INDArray, input1: INDArray): ForwardPass[Cache2, INDArray] = ???
      override def backward2(cache2: Cache2, patchOutput: Patch[INDArray]) = ???
    }

    object Exp extends Bifunction[INDArray, INDArray] {
      override type Cache = INDArray

      override def backward(cache: Cache, patchOutput: Patch[INDArray]): Patch[INDArray] = {
        patchOutput match {
          case Delta(deltaOutput) =>
            Delta(deltaOutput * cache)
        }
      }

      override def forward(input: INDArray) = new ForwardPass[Cache, INDArray] {
        override def output: INDArray = {
          cache
        }

        override lazy val cache: INDArray = {
          Transforms.exp(input)
        }
      }
    }


    final case class Compose[A, B, C](f: Bifunction[B, C], g: Bifunction[A, B]) extends Bifunction[A, C] {

      trait Cache {
        def cacheF: f.Cache

        def cacheG: g.Cache
      }

      override def backward(cache: Cache, patchOutput: Patch[C]): Patch[A] = {
        g.backward(cache.cacheG, f.backward(cache.cacheF, patchOutput))
      }

      override def forward(input: A) = new ForwardPass[Cache, C] {

        lazy val forwardPassG = g.forward(input)
        lazy val forwardPassF = f.forward(forwardPassG.output)

        override def output: C = forwardPassF.output

        override def cache: Cache = new Cache {

          override def cacheG: g.Cache = forwardPassG.cache

          override def cacheF: f.Cache = forwardPassF.cache
        }
      }
    }

    final case class Substitute[A, B, C](x: Bifunction[A, Bifunction[B, C]], y: Bifunction[A, B]) extends Bifunction[A, C] {

      trait Cache {
        def cacheX: x.Cache

        def cacheY: y.Cache

        def cacheXY: bifunctionXY.Cache

        def forwardXY: ForwardPass[bifunctionXY.Cache, C]

        val bifunctionXY: Bifunction[B, C]
      }

      override def backward(cache: Cache, patchC: Patch[C]): Patch[A] = {
        val patchB: Patch[B] = cache.bifunctionXY.backward(cache.cacheXY, patchC)
        y.backward(cache.cacheY, patchB).merge(x.backward(cache.cacheX, CurriedPatch(patchB, patchC)))
      }

      override def forward(input: A) = new ForwardPass[Cache, C] {
        thisForwardPass =>

        override def output: C = {
          cache.forwardXY.output
        }

        override def cache: Cache = new Cache {

          private lazy val forwardY = y.forward(input)

          private lazy val forwardX = x.forward(input)

          override lazy val bifunctionXY = forwardX.output

          override def forwardXY = bifunctionXY.forward(forwardY.output)

          override def cacheX = forwardX.cache

          override def cacheXY = forwardXY.cache

          override def cacheY = forwardY.cache

        }
      }
    }


    implicit object BifunctionInstances extends SKICombinator[Bifunction] with Exp[Bifunction] {

      override def exp = Exp


      override def id[A] = Bifunction.Id[A]()


      override def compose[A, B, C](f: Bifunction[B, C], g: Bifunction[A, B]) = Compose(f, g)

      override def substitute[A, B, C](x: Bifunction[A, Bifunction[B, C]], y: Bifunction[A, B]): Bifunction[A, C] = Substitute(x, y)

      override def constant[A, B, C](x: Bifunction[A, B]): Bifunction[C, Bifunction[A, B]] = ???
    }

  }


  //
  //  trait DotFactory[=>:[_, _]] {
  //    def dot: INDArray =>: INDArray =>: INDArray
  //  }
  //
  //  trait ExpFactory[=>:[_, _]] {
  //    def exp: INDArray =>: INDArray
  //  }
  //
  //  trait AdditionFactory[=>:[_, _]] {
  //    def add: INDArray =>: INDArray =>: INDArray
  //  }
  //
  //  trait ForwardPass[+Output, +Cache] {
  //
  //    def output: Output
  //
  //    def cache: Cache
  //
  //  }
  //
  //  sealed trait Forward[-Input, +Output] {
  //
  //    type Cache
  //
  //    def forward(input: Input): ForwardPass[Output, Cache]
  //
  //  }
  //
  //  final case class PartiallyAppliedForward[Input1, -Input2, +Output, UnderlyingCache](input1: Input1, curriedForward: Forward2[Input1, Input2, Output] {
  //    type Cache2 = UnderlyingCache
  //  }) extends Forward[Input2, Output] {
  //    type Cache = UnderlyingCache
  //
  //    override def forward(input2: Input2): ForwardPass[Output, Cache] = {
  //      curriedForward.forward2(input1, input2)
  //    }
  //  }
  //
  //  sealed trait Forward1[-Input, +Output] extends Forward[Input, Output] {
  //    type ForwardPass = DeepLearning.ForwardPass[Output, Cache]
  //  }
  //
  //  sealed trait Forward2[-Input1, -Input2, +Output] extends Forward[Input1, Forward[Input2, Output]] {
  //
  //    final type Cache = Unit
  //
  //    type Cache2
  //
  //    final def forward(input: Input1) = new DeepLearning.ForwardPass[PartiallyAppliedForward[_, Input2, Output, Cache2], Cache] {
  //      override def output: PartiallyAppliedForward[_, Input2, Output, Cache2] = {
  //        PartiallyAppliedForward[Input1, Input2, Output, Cache2](input, Forward2.this)
  //      }
  //
  //      override def cache: Unit = ()
  //    }
  //
  //    def forward2(input1: Input1, input2: Input2): DeepLearning.ForwardPass[Output, Cache2]
  //
  //    type ForwardPass = DeepLearning.ForwardPass[Output, Cache2]
  //
  //  }
  //
  //
  //  implicit object Forward extends Split[Forward] with Category[Forward] with DotFactory[Forward] with AdditionFactory[Forward] with ExpFactory[Forward] {
  //
  //    case object dot extends Forward2[INDArray, INDArray, INDArray] {
  //      def forward2(input1: INDArray, input2: INDArray) = new ForwardPass {
  //        override def output: INDArray = ???
  //
  //        override def cache: Cache2 = ???
  //      }
  //    }
  //
  //    case object add extends Forward2[INDArray, INDArray, INDArray] {
  //
  //      def forward2(input1: INDArray, input2: INDArray) = new ForwardPass {
  //        override def output: INDArray = ???
  //
  //        override def cache: Cache2 = ???
  //      }
  //    }
  //
  //    case object exp extends Forward1[INDArray, INDArray] {
  //
  //      type Cache = Unit
  //
  //      def forward(input: INDArray) = new ForwardPass {
  //        override def output: INDArray = ???
  //
  //        override def cache: Cache = ???
  //      }
  //    }
  //
  //    override def split[A, B, C, D](f: Forward[A, B], g: Forward[C, D]): Forward[(A, C), (B, D)] = ???
  //
  //    override def id[A]: Forward[A, A] = ???
  //
  //    override def compose[A, B, C](f: Forward[B, C], g: Forward[A, B]): Forward[A, C] = ???
  //  }
  //
  //
  //  trait Patch[Weight] {
  //    def apply(oldWeight: Weight): Weight
  //  }
  //
  //  trait Update[Weight, Patch] {
  //    def updated(oldWeight:Weight, patch: Patch):Weight
  //  }
  //
  //  sealed trait Backward[Input, Output] {
  //
  //    //    type BackwardPassInput = F[Output]
  //    //
  //    //    type BackwardPassOutput = F[Input]
  //
  //    type Cache
  //
  //    def backward[BackwardPassInput, BackwardPassOutput](cache: Cache, backwardPassInput: BackwardPassInput)
  //                                                       (implicit outputDifferentiable: Update[Output, BackwardPassInput],
  //                                                        inputDifferentiable: Update[Input, BackwardPassOutput]
  //                                                       ): BackwardPassOutput
  //
  //    def forward: Forward[Input, Output] {type Cache = Backward.this.Cache}
  //
  //  }
  //
  //  implicit object Backward extends DotFactory[Backward] with AdditionFactory[Backward] with Split[Backward] with Choice[Backward] {
  //
  //    case object exp extends Backward[INDArray, INDArray] {
  //      override type BackwardPassInput = INDArray
  //
  //      override def forward: Forward.exp.type = Forward.exp
  //
  //      override def backward(cache: Unit, backwardPassInput: INDArray): INDArray = ???
  //
  //      override type Cache = Unit
  //      override type BackwardPassOutput = INDArray
  //    }
  //
  //    //    final case class Dot(left: Backward[INDArray], right: Backward[INDArray]) extends Backward[INDArray] {
  //    //      override def forward: Forward.Dot = Forward.Dot(left.forward, right.forward)
  //    //    }
  //    //
  //    //    final case class Plus(left: Backward[INDArray], right: Backward[INDArray]) extends Backward[INDArray] {
  //    //      override def forward: Forward.Plus = Forward.Plus(left.forward, right.forward)
  //    //    }
  //    //
  //    //    override def ap[A, B](fa: Backward[A])(f: Backward[(A) => B]): Backward[B] = Ap(fa, f)
  //    //
  //    //    final case class Ap[A, B](fa: Backward[A], f: Backward[(A) => B]) extends Backward[B] {
  //    //      override def forward: Forward[B] = Forward.Ap(fa.forward, f.forward)
  //    //    }
  //    //
  //    //
  //    override def compose[A, B, C](f: Backward[B, C], g: Backward[A, B]): Backward[A, C] = ???
  //
  //
  //  }
  //

  /*


  Backward.compose(Backward.exp, Backward.exp)

    import Backward
    import scalaz.syntax.compose._

    def f(implicit xxx) = {
      exp compose exp
    }

   */

  //
  //  trait NeuronNetwork {
  //    type Out
  //  }
  //
  //  object NeuronNetwork {
  //
  //    type Aux[Out0] = NeuronNetwork {
  //      type Out = Out0
  //    }
  //
  //  }
  //
  //  trait Forward[NN <: NeuronNetwork, In] extends DepFn2[NN, In] {
  //    type In
  //    type Cache
  //  }
  //
  //  object Forward {
  //
  //    implicit def forward[NN <: NeuronNetwork, In] = new Forward[NN, In] {
  //      override type In = this.type
  //      override type Cache = this.type
  //
  //      override def apply(t: NN, u: In): Out = ???
  //
  //      override type Out = (Cache)
  //    }
  //
  //  }

  //
  //    trait Forward[Expression] extends DepFn[Expression] {
  //
  //      type ForwardPassInput
  //      type ForwardPassOutput
  //
  //      trait ForwardPass {
  //        def output: ForwardPassOutput
  //
  //        def cache: Cache
  //      }
  //
  //      type Cache
  //
  //      def forward(layer: Layer, input: ForwardPassInput): ForwardPass
  //
  //    }
  //
  //  object Forward {
  //
  //
  //
  //  }

  //  final case class Exp[Upstream <: NeuronNetwork](upstream: Upstream) extends NeuronNetwork.Aux[INDArray]
  //
  //  final case class Select[Upstream <: NeuronNetwork, Field](upstream: Upstream) extends NeuronNetwork.Aux[Field]

  //  implicit class PredictOps[NN <: NeuronNetwork](neuronNetwork: NN)(implicit input: Input[NN]) {
  //
  //    def predict(input: input.Out): neuronNetwork.Out = {
  //      ???
  //    }
  //
  //  }


  // 整个网络就是一个递归的 NeuronNetwork,不包含其他节点
  // 这个 NeuronNetwork 有类型

  //  def train(): Unit = {
  //
  //  }


  //  trait Layer[Input <: HList, Output] extends PolyApply {
  //    def forward(input: Input): Output
  //  }

  //  case class NeuronNetwork

  //
  //  trait NeuronNetwork {
  //    type Input
  //    type Output
  //  }
  //
  //  trait ForwardPass[NN <: NeuronNetwork] {
  //
  //    def forward
  //
  //  }
  //
  //  final case class FullyConnected(data: INDArray) extends NeuronNetwork {
  //
  //    type Input = INDArray
  //
  //    type Output = INDArray
  //
  //    override def forward(input: Input): Output = {
  //      input dot data
  //    }
  //
  //  }
  //
  //  trait Backpropagation[NN <: NeuronNetwork] {
  //
  //    type BackwardPassOutput
  //    type BackwardPassInput
  //
  //    trait BackwardPass {
  //      def output: BackwardPassOutput
  //
  //      def patch: Patch
  //    }
  //
  //    type Patch
  //
  //    def backward[Cache](neuronNetwork: NN {type Output <: Cache}, cache: Cache, backwardPassInput: BackwardPassInput): BackwardPassOutput
  //
  //  }
  //
  //
  //  //
  //  //  trait Forward {
  //  //
  //  //    type Output
  //  //
  //  //    type Cache
  //  //
  //  //    def output: Output
  //  //
  //  //    def cache: Cache
  //  //
  //  //  }
  //  //
  //  //  trait Backward {
  //  //
  //  //    type Output
  //  //
  //  //    type Patch
  //  //
  //  //    def output: Output
  //  //
  //  //    def patch: Patch
  //  //
  //  //  }
  //  //
  //  //  // TODO: Loss
  //  //  //  final class Loss(val value: Double) extends AnyVal
  //  //
  //  //  implicit final class WeightOps[Weight](weight: Weight) {
  //  //
  //  //    def train[Input, Cache0, ExpectedOutput, Patch0]
  //  //    (input: Input, expectedOutput: ExpectedOutput)
  //  //    (
  //  //      implicit
  //  //      forward: (Weight, Input) => Forward {type Cache <: Cache0},
  //  //      backward: (Cache0, ExpectedOutput) => Backward {type Patch = Patch0}
  //  //    ): Patch0 = {
  //  //      backward(forward(weight, input).cache, expectedOutput).patch
  //  //    }
  //  //
  //  //    def predict[Input, Output0](input: Input)(implicit forward: (Weight, Input) => Forward {type Output = Output0}): Output0 = {
  //  //      forward(weight, input).output
  //  //    }
  //  //
  //  //    //
  //  //    //    def loss[Input, Cache0, ExpectedOutput]
  //  //    //    (input: Input, expectedOutput: ExpectedOutput)
  //  //    //    (
  //  //    //      implicit
  //  //    //      forward: (Weight, Input) => Forward {type Cache <: Cache0},
  //  //    //      loss: (Cache0, ExpectedOutput) => Double
  //  //    //    ) = {
  //  //    //      loss(forward(weight, input).cache, expectedOutput)
  //  //    //    }
  //  //
  //  //  }
  //  //
  //  //
  //  //  //
  //  //  //  trait Forward {
  //  //  //    type Input
  //  //  //    type Output
  //  //  //    type Weight
  //  //  //
  //  //  //    def forward(weight: Weight, input: Input): Output
  //  //  //  }
  //  //  //
  //  //  //  trait Backward {
  //  //  //    type Input
  //  //  //    type Output
  //  //  //    type Cache
  //  //  //
  //  //  //  }
  //

}