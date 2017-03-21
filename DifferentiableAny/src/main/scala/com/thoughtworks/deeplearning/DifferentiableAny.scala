package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.DifferentiableAny.Layers.{Compose, WithOutputDataHook}
import com.thoughtworks.deeplearning.Layer.Tape
import com.thoughtworks.deeplearning.Symbolic.Layers.Literal
import com.thoughtworks.deeplearning.Symbolic._
import resource.managed

import language.implicitConversions
import language.existentials

/**
  * A namespace of common operators for any layers.
  *
  * After importing `DifferentiableAny._`, the following methods will be available on any layers.
  *  - [[DifferentiableAny.AnyLayerOps.compose compose]]
  *  - [[DifferentiableAny.AnyLayerOps.predict predict]]
  *  - [[DifferentiableAny.AnyLayerOps.train train]]
  *  - [[DifferentiableAny.AnyLayerOps.withOutputDataHook withOutputDataHook]]
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableAny {

  private[deeplearning] type AnyPlaceholder = Placeholder[Any, ExistentialNothing]
  private[deeplearning] val AnyPlaceholder: AnyPlaceholder = implicitly

  object Layers {

    final case class Compose[Input0 <: Tape, Temporary <: Tape, Output0 <: Tape](
        leftOperand: Layer.Aux[Temporary, Output0],
        rightOperand: Layer.Aux[Input0, Temporary])
        extends Layer {
      override type Input = Input0
      override type Output = Output0

      override def forward(input: Input): Output = {
        val tmpTape = rightOperand.forward(input)
        try {
          leftOperand.forward(tmpTape)
        } finally {
          tmpTape.close()
        }
      }
    }

    final case class WithOutputDataHook[Input0 <: Tape, OutputData, OutputDelta](
        anyLayer: Layer.Aux[Input0, Tape.Aux[OutputData, OutputDelta]],
        hook: OutputData => Unit)
        extends Layer {
      override type Input = Input0
      override type Output = Tape.Aux[OutputData, OutputDelta]

      override def forward(input: Input): Output = {
        val output = anyLayer.forward(input)
        hook(output.value)
        output
      }
    }

  }

  /**
    * A helper that contains common operations for any layers.
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableAny._
    * (input:INDArray @Symbolic).compose(anotherLayer)
    * }}}
    */
  final class AnyLayerOps[Input <: Tape, OutputData, OutputDelta](
      anyLayer: Layer.Aux[Input, Tape.Aux[OutputData, OutputDelta]]) {

    /**
      * Returns a [[Layer]] that accepts `g`'s output as input of `anyLayer`.
      *
      * @usecase def compose[NewInput <: Tape](g: Layer.Aux[NewInput, Input]): Layer.Aux[NewInput, Tape.Aux[OutputData, OutputDelta]] = ???
      *
      * @example{{{
      * import com.thoughtworks.deeplearning.DifferentiableAny._
      * def composeNetwork(implicit thisLayer: INDArray @Symbolic)(anotherLayer: INDArray @Symbolic) = {
      *   thisLayer.compose(anotherLayer)
      * }}}
      */
    def compose[G, NewInput <: Tape, InputData, InputDelta](g: G)(
        implicit toLayer: ToLayer.Aux[G, NewInput, InputData, InputDelta],
        toInput: Layer.Aux[NewInput, Tape.Aux[InputData, InputDelta]] <:< Layer.Aux[NewInput, Input]
    ): Layer.Aux[NewInput, Tape.Aux[OutputData, OutputDelta]] = {
      Compose[NewInput, Input, Tape.Aux[OutputData, OutputDelta]](anyLayer, toInput(toLayer(g)))
    }

    /**
      * Returns the result of inputData's [[Layer.forward forward]].
      *
      * @example{{{
      * import com.thoughtworks.deeplearning.DifferentiableAny._
      * def composeNetwork(implicit inputData: INDArray @Symbolic) = ???
      * val predictor = composeNetwork
      * predictor.predict(testData)
      * }}}
      */
    def predict[InputData, InputDelta](inputData: InputData)(
        implicit ev: Layer.Aux[Input, Tape.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
          Tape.Aux[InputData, InputDelta],
          Tape.Aux[OutputData, OutputDelta]]
    ): OutputData = {
      managed(anyLayer.forward(Literal[InputData](inputData))).acquireAndGet(_.value)
    }

    /**
      * Updates those weights embedded in `anyLayer` according to the result of inputData's [[com.thoughtworks.deeplearning.Layer.Tape#backward backward]].
      *
      * @example{{{
      * import com.thoughtworks.deeplearning.DifferentiableAny._
      * def composeNetwork(implicit input: INDArray @Symbolic) = ???
      * val yourNetwork=composeNetwork
      * yourNetwork.train(testData)
      * }}}
      */
    def train[InputData, InputDelta](inputData: InputData)(
        implicit ev: Layer.Aux[Input, Tape.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
          Tape.Aux[InputData, InputDelta],
          Tape.Aux[OutputData, OutputDelta]],
        outputDataIsOutputDelta: Trainable[OutputData, OutputDelta]
    ): OutputData = {
      val outputTape = anyLayer.forward(Literal[InputData](inputData))
      try {
        val loss = outputTape.value
        outputTape.backward(outputDataIsOutputDelta(loss))
        loss
      } finally {
        outputTape.close()
      }

    }

    /**
      * Returns a new [[Layer]] which is a wrapper of result of `anyLayer`'s [[Layer.forward forward]] and will invoke `hook(anyLayer.forward(input).value)`.
      * In DeepLearning.Scala, operation is not immediately run,
      * but first filled the network with placeholders, until the entire network is running, the real data will replace placeholders.
      * So if you want to know some layers's intermediate state, you need to use `withOutputDataHook`.
      *
      * @example{{{
      * import com.thoughtworks.deeplearning.DifferentiableAny._
      * (var:INDArray @Symbolic).withOutputDataHook{ data => println(data) }
      * }}}
      */
    def withOutputDataHook(hook: OutputData => Unit): Layer.Aux[Input, Tape.Aux[OutputData, OutputDelta]] = {
      WithOutputDataHook(anyLayer, hook)
    }
  }

  /**
    * Implicitly converts any layer to [[AnyLayerOps]], which enables common methods for any layers.
    *
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableAny._
    * }}}
    */
  implicit def toAnyLayerOps[A, Input <: Tape, OutputData, OutputDelta](a: A)(
      implicit toLayer: ToLayer.Aux[A, Input, OutputData, OutputDelta])
    : AnyLayerOps[Input, OutputData, OutputDelta] = {
    new AnyLayerOps(toLayer(a))
  }

  type ExistentialNothing = T forSome { type T >: Nothing <: Nothing }

  implicit def anyToLiteral: ToLiteral.Aux[Any, Any, ExistentialNothing] = ToLiteral.fromData

  /**
    * A type class that makes result of [[Layer.forward forward]] as input of [[[[com.thoughtworks.deeplearning.Layer.Tape#backward backward]]]].
    * To train a layer, a implement of `Trainable` has parameterized with types of the layer's [[Layer.Input Input]] and [[Layer.Output Output]] is required.
    *
    * @see This type class is required by [[AnyLayerOps.train train]].
    */
  trait Trainable[-Data, +Delta] {
    def apply(data: Data): Delta
  }

}
