package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Tape
import com.thoughtworks.deeplearning.Symbolic.Layers.Literal
import shapeless._

import scala.annotation.implicitNotFound
import scala.language.{existentials, implicitConversions}

/**
  * Provides `@Symbolic` annotation to create '''[[https://en.wikipedia.org/wiki/Symbolic_computation symbolic]] methods''', in which you can create [[Layer]]s from mathematical formulas.
  *
  * `Symbolic` is a [[https://en.wikipedia.org/wiki/Dependent_type dependent]] [[https://en.wikipedia.org/wiki/Type_class type class]] that calculates a specific [[Layer]] type according to `NativeOutput`.
  * Combining with [[https://github.com/ThoughtWorksInc/implicit-dependent-type implicit-dependent-type]] compiler plugin,
  * it can be treated as a type [[http://www.scala-lang.org/files/archive/spec/2.12/11-annotations.html annotation]] in the form of `NativeOutput @Symbolic`, converting `NativeOutput` to a specific [[Layer]] type.
  *
  * == Three usages of `@Symbolic` ==
  *
  * === Implicit parameter types used for symbol methods ===
  *
  * In case that the implicit parameter of an method is marked with `@Symbolic`, then this method is symbol method. The implicit parameter type marked with `@Symbolic` is the '''input type''' of this symbol method.
  * In this case, `NativeOutput @Symbolic` will be expanded as:
  * {{{Identity[NativeOutput, Derivative type of NativeOutput]}}}
  *
  * For example:
  *
  * {{{
  * def sumNetwork(implicit scores: INDArray @Symbolic): Double = {
  *   exp(scores).sum
  * }
  * }}}
  *
  * In the above code, because [[Layer.Tape#Delta the derivative type]] of `INDArray` is also `INDArray`, the input type `INDArray @Symbolic` of `sumNetwork`, once expanded, is `Identity[INDArray, INDArray]`
  *
  * === Used for symbol method internal variable and return value ===
  *
  * A `NativeOutput @Symbolic` inside a symbol method, or at the return position of a symbol method, will be expanded as:
  * {{{Layer.Aux[Tape.Aux[value type of input type, derivative type of input type], Tape.Aux[NativeOutput, derivative type of NativeOutput]]}}}
  *
  * For example:
  *
  *
  * {{{
  * def sumNetwork(implicit scores: INDArray @Symbolic): Double @Symbolic = {
  *   val expScores: INDArray @Symbolic = exp(scores)
  *   val result: Double @Symbolic = expScores.sum
  *   result
  * }
  * }}}
  *
  * In the above code, the type `INDArray @Symbolic` of `expScores` is expanded as:
  * {{{Layer.Aux[Tape.Aux[INDArray, INDArray], Tape.Aux[INDArray, INDArray]]}}}
  *
  * The type `Double @Symbolic` of `result` is expanded as:
  * {{{Layer.Aux[Tape.Aux[INDArray, INDArray], Tape.Aux[Double, Double]]}}}
  *
  * === Used for cases excluding symbol method ===
  *
  * `(NativeInput => NativeOutput) @Symbolic` outside a symbol method, will be expanded as:
  * {{{Layer.Aux[Tape.Aux[NativeInput, derivative type of NativeInput], Tape.Aux[NativeOutput, derivative type of NativeOutput]]}}}
  *
  * For example:
  *
  * {{{
  * val predictor: (INDArray => Double) @Symbolic = sumNetwork
  * }}}
  *
  * In the above code, type `(INDArray => Double) @Symbolic` of `predictor` is expanded as:
  *
  * {{{Layer.Aux[Tape.Aux[INDArray, INDArray], Tape.Aux[Double, Double]]}}}
  *
  * == Custom symbol type ==
  *
  *
  * The `@Symbolic` determines the mapping relation between the primitive type and derivative by checking [[Symbolic.ToLiteral]] implicit value. Therefore, `@Symbolic` can be a custom symbol type once you define your own the implicit [[Symbolic.ToLiteral]].
  *
  * For example, if you want to support `Short @Symbolic`, using [[scala.Float Float]] as the derivative type of [[scala.Short Short]], then you can conduct the follows:
  *
  *
  * {{{
  * implicit object ShortToLiteral extends ToLiteral[Short] {
  *   override type Data = Short
  *   override type Delta = Float
  *   override def apply(data: Short) = Literal(data)
  * }
  *
  * def makeShortNetwork(implicit input: Short @Symbolic): Short @Symbolic = {
  *   input
  * }
  *
  * val shortNetwork: (Short => Short) @Symbolic = makeShortNetwork
  * }}}
  *
  * Thus, type of `shortNetwork` is expanded as:
  * {{{Layer.Aux[Tape.Aux[Short, Float], Tape.Aux[Short, Float]]}}}
  * @see [[Symbolic.Layers.Identity]]
  * @see [[Symbolic.ToLiteral]]
  * @see [[Layer.Tape#Delta]]
  *
  */
@implicitNotFound("Don't know how to make ${NativeOutput} differentiable")
trait Symbolic[NativeOutput] {
  type `@` <: Layer
}

private[deeplearning] trait LowPrioritySymbolic { this: Symbolic.type =>

  implicit def from[NativeOutput, Data0, Delta0](
      implicit toLiteral: Lazy[ToLiteral.Aux[NativeOutput, Data0, Delta0]]): From.Aux[NativeOutput, Data0, Delta0] =
    new From[NativeOutput] {
      type Data = Data0
      type Delta = Delta0
    }

}

/**
  * There are two ways to convert a value to [[Layer]].
  *
  * The first way is invoke [[Symbolic.ToLayer toLayer]], such as:
  * {{{
  * def createMyNeuralNetwork(implicit input: Float @Symbolic): Float @Symbolic = {
  *   val floatLayer: Float @Symbolic = 1.0f.toLayer
  *   floatLayer
  * }
  * }}}
  *
  * The second way is [[Symbolic.autoToLayer autoToLayer]], such as:
  * {{{
  * def createMyNeuralNetwork(implicit input: Float @Symbolic): Float @Symbolic = {
  *   val floatLayer: Float @Symbolic = 1.0f
  *   floatLayer
  * }
  * }}}
  *
  * In order to compile the above code through, you will need:
  *
  * {{{
  * import com.thoughtworks.deeplearning.Symbolic._
  * import com.thoughtworks.deeplearning.Symbolic
  * import com.thoughtworks.deeplearning.DifferentiableFloat._
  * }}}
  *
  */
object Symbolic extends LowPrioritySymbolic {

  trait ToLiteral[From] extends DepFn1[From] {

    type Data
    type Delta

    type Out = Literal[Data]

  }

  object ToLiteral {

    def fromData[From <: Data0, Data0, Delta0] = new ToLiteral[From] {
      override type Data = Data0
      override type Delta = Delta0

      override def apply(data: From) = Literal[Data](data)
    }

    type Aux[From, Data0, Delta0] = ToLiteral[From] {
      type Data = Data0
      type Delta = Delta0
    }

  }

  object Layers {

    final case class Identity[Data0, Delta0]() extends Layer {

      type Data = Data0
      type Delta = Delta0

      type Input = Tape.Aux[Data, Delta]
      type Output = Tape.Aux[Data, Delta]

      override def forward(input: Input): Output = {
        input.duplicate()
      }

      private type ConcreteTape = Tape.Aux[Data, Delta]

      // Workaround for https://issues.scala-lang.org/browse/SI-10008
      type Tape >: ConcreteTape <: ConcreteTape

      private[deeplearning] type To[OutputPlaceholder <: Identity[_, _]] = Layer.Aux[Tape, OutputPlaceholder#Tape]

    }

    object Identity {

      implicit def implicitlyApply[Data, Delta]: Identity[Data, Delta] = new Identity

      private[deeplearning] type DataOf[`@` <: Identity[_, _]] = t.Data forSome { val t: `@` }
      private[deeplearning] type DeltaOf[`@` <: Identity[_, _]] = t.Delta forSome { val t: `@` }

      implicit def inputPlaceholderToLayer[InputData, InputDelta]
        : ToLayer.Aux[Identity[InputData, InputDelta], Tape.Aux[InputData, InputDelta], InputData, InputDelta] =
        new ToLayer[Identity[InputData, InputDelta], Tape.Aux[InputData, InputDelta]] {
          override type OutputData = InputData
          override type OutputDelta = InputDelta

          override def apply(input: Identity[InputData, InputDelta]) =
            Identity[InputData, InputDelta]()
        }

    }

    final case class Literal[Data0](value0: Data0) extends Layer with Tape {
      override type Data = Data0
      override type Delta = Any
      override type Input = Tape
      override type Output = Tape.Aux[Data, Delta]

      override def value: Data = value0

      override def forward(input: Input) = this

      override def isTrainable: Boolean = false

      override protected def forceBackward(delta: Delta): Unit = {}

      override def close(): Unit = {}

      override def duplicate() = this
    }

  }

  import Layers._

  private[deeplearning] trait IsLayer {
    type OutputData
    type OutputDelta
    type InputData
    type InputDelta
    type ConcreteLayer = Layer.Aux[Tape.Aux[InputData, InputDelta], Tape.Aux[OutputData, OutputDelta]]
    type `@` >: ConcreteLayer <: ConcreteLayer
  }

  private[deeplearning] object IsLayer {

    type Aux[InputData0, InputDelta0, OutputData0, OutputDelta0] = IsLayer {
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
      type InputData = InputData0
      type InputDelta = InputDelta0
    }

  }

  private[deeplearning] trait To[NativeOutput] extends Symbolic[NativeOutput] with IsLayer

  private[deeplearning] object To {
    type Aux[NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0] = To[NativeOutput] {
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
      type InputData = InputData0
      type InputDelta = InputDelta0
    }

    def apply[NativeOutput](implicit tc: To[NativeOutput]): tc.type = tc
  }

  implicit def to[NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0](
      implicit inputPlaceHolder: Identity[InputData0, InputDelta0],
      toLiteral: ToLiteral.Aux[NativeOutput, OutputData0, OutputDelta0]
  ): To.Aux[NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0] =
    new To[NativeOutput] {
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
      type InputData = InputData0
      type InputDelta = InputDelta0
    }

  private[deeplearning] trait FromTo[NativeInput, NativeOutput]
      extends Symbolic[NativeInput => NativeOutput]
      with IsLayer

  private[deeplearning] object FromTo {

    /** @template */
    type Aux[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0] =
      FromTo[NativeInput, NativeOutput] {
        type InputData = InputData0
        type InputDelta = InputDelta0
        type OutputData = OutputData0
        type OutputDelta = OutputDelta0
      }

    def apply[NativeInput, NativeOutput](implicit typeClass: FromTo[NativeInput, NativeOutput]): typeClass.type =
      typeClass

  }

  implicit def fromTo[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0](
      implicit inputToLiteral: Lazy[ToLiteral.Aux[NativeInput, InputData0, InputDelta0]],
      outputToLiteral: Lazy[ToLiteral.Aux[NativeOutput, OutputData0, OutputDelta0]])
    : FromTo.Aux[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0] =
    new FromTo[NativeInput, NativeOutput] {
      type InputData = InputData0
      type InputDelta = InputDelta0
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
    }

  private[deeplearning] type Placeholder[Data, Delta] = Identity[Data, Delta]

  private[deeplearning] val Placeholder = Identity

  implicit final class ToLayerOps[From, Input <: Tape, OutputData, OutputDelta](from: From)(
      implicit typeClassInstance: ToLayer.Aux[From, Input, OutputData, OutputDelta]
  ) {

    def toLayer: Layer.Aux[Input, Tape.Aux[OutputData, OutputDelta]] = typeClassInstance(from)

  }

  implicit final class ToTapeOps[From, Data, Delta](from: From)(
      implicit lift: ToLiteral.Aux[From, Data, Delta]
  ) {

    @inline
    def toTape: Tape.Aux[Data, Delta] = lift(from)

  }

  implicit def autoToLayer[A, Input <: Tape, OutputData, OutputDelta](a: A)(
      implicit toLayer: ToLayer.Aux[A, Input, OutputData, OutputDelta])
    : Layer.Aux[Input, Tape.Aux[OutputData, OutputDelta]] = {
    toLayer(a)
  }

  private[deeplearning] sealed trait ToLayerLowPriorityImplicits { this: ToLayer.type =>

    implicit def toLayerOfPlaceholder[Input0 <: Tape, OutputPlaceholder <: Identity[_, _]]
      : ToLayer.OfPlaceholder[Layer.Aux[Input0, OutputPlaceholder#Tape], Input0, OutputPlaceholder] = {
      ToLayer
        .layerToLayer[Input0, Placeholder.DataOf[OutputPlaceholder], Placeholder.DeltaOf[OutputPlaceholder]]
        .asInstanceOf[ToLayer.OfPlaceholder[Layer.Aux[Input0, OutputPlaceholder#Tape], Input0, OutputPlaceholder]]
    }

    implicit def isLayerToLayer[NativeInput, NativeOutput, InputData0, InputDelta0, OutputData0, OutputDelta0]
      : ToLayer.Aux[
        IsLayer.Aux[InputData0, InputDelta0, OutputData0, OutputDelta0]#`@`,
        Tape.Aux[InputData0, InputDelta0],
        OutputData0,
        OutputDelta0
      ] = {
      layerToLayer
    }

  }

  object ToLayer extends ToLayerLowPriorityImplicits {

    type Aux[From, Input <: Tape, OutputData0, OutputDelta0] = ToLayer[From, Input] {
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
    }

    type OfPlaceholder[From, Input <: Tape, OutputPlaceholder <: Identity[_, _]] =
      ToLayer.Aux[From, Input, differentiablePlaceholder.Data, differentiablePlaceholder.Delta] forSome {
        val differentiablePlaceholder: OutputPlaceholder
      }

    implicit def layerToLayer[Input <: Tape, OutputData0, OutputDelta0]
      : ToLayer.Aux[Layer.Aux[Input, Tape.Aux[OutputData0, OutputDelta0]], Input, OutputData0, OutputDelta0] =
      new ToLayer[Layer.Aux[Input, Tape.Aux[OutputData0, OutputDelta0]], Input] {
        override type OutputData = OutputData0
        override type OutputDelta = OutputDelta0

        override def apply(layer: Layer.Aux[Input, Tape.Aux[OutputData, OutputDelta]]) = layer
      }

    implicit def placeholderToLayer[From, InputData, InputDelta, OutputData0, OutputDelta0](
        implicit inputPlaceholder: Identity[InputData, InputDelta],
        toLiteral: ToLiteral.Aux[From, OutputData0, OutputDelta0])
      : ToLayer.Aux[From, Tape.Aux[InputData, InputDelta], OutputData0, OutputDelta0] = {
      new ToLayer[From, Tape.Aux[InputData, InputDelta]] {
        override type OutputData = OutputData0
        override type OutputDelta = OutputDelta0

        override def apply(from: From) = toLiteral(from)
      }
    }

  }

  /**
    * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
    */
  @implicitNotFound("Cannot convert ${From} to layer")
  trait ToLayer[From, Input <: Tape] extends DepFn1[From] {
    type OutputData
    type OutputDelta
    type Output = Tape.Aux[OutputData, OutputDelta]
    type Out = Layer.Aux[Input, Output]
  }

  private[deeplearning] trait From[NativeOutput] extends Symbolic[NativeOutput] with DepFn0 {

    type Data
    type Delta

    type `@` = Identity[Data, Delta]

    type Out = `@`

    override def apply() = new Identity

  }

  private[deeplearning] object From {

    /** @template */
    type Aux[NativeOutput, Data0, Delta0] = From[NativeOutput] {
      type Data = Data0
      type Delta = Delta0
    }

    def apply[NativeOutput](implicit typeClass: From[NativeOutput]): typeClass.type = typeClass

  }

}
