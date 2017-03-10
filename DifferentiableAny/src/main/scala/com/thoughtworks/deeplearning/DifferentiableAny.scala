package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.DifferentiableAny.Layers.{Compose, WithOutputDataHook}
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Symbolic.Layers.Literal
import com.thoughtworks.deeplearning.Symbolic._
import resource.managed

import language.implicitConversions
import language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableAny {

  private[deeplearning] type AnyPlaceholder = Placeholder[Any, ExistentialNothing]
  private[deeplearning] val AnyPlaceholder: AnyPlaceholder = implicitly

  object Layers {

    final case class Compose[Input0 <: Batch, Temporary <: Batch, Output0 <: Batch](
        leftOperand: Layer.Aux[Temporary, Output0],
        rightOperand: Layer.Aux[Input0, Temporary])
        extends Layer {
      override type Input = Input0
      override type Output = Output0

      override def forward(input: Input): Output = {
        val tmpBatch = rightOperand.forward(input)
        try {
          leftOperand.forward(tmpBatch)
        } finally {
          tmpBatch.close()
        }
      }
    }

    final case class WithOutputDataHook[Input0 <: Batch, OutputData, OutputDelta](
        layer: Layer.Aux[Input0, Batch.Aux[OutputData, OutputDelta]],
        hook: OutputData => Unit)
        extends Layer {
      override type Input = Input0
      override type Output = Batch.Aux[OutputData, OutputDelta]

      override def forward(input: Input): Output = {
        val output = layer.forward(input)
        hook(output.value)
        output
      }
    }

  }

  /**
    * A helper that contains common ops for all layers
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableAny._
    * (input:From[INDArray]##T).compose(anotherLayer)
    * }}}
    */
  final class AnyLayerOps[Input <: Batch, OutputData, OutputDelta](
      layer: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) {

    /**
      * Returns a [[Layer]] that accepts another layer's output as input of this layer
      * @example{{{
      * import com.thoughtworks.deeplearning.DifferentiableAny._
      * def composeNetwork(implicit thisLayer: Symbolic[INDArray]##T)(anotherLayer: Symbolic[INDArray]##T) = {
      *   thisLayer.compose(anotherLayer)
      * }}}
      */
    def compose[G, NewInput <: Batch, InputData, InputDelta](g: G)(
        implicit toLayer: ToLayer.Aux[G, NewInput, InputData, InputDelta],
        toInput: Layer.Aux[NewInput, Batch.Aux[InputData, InputDelta]] <:< Layer.Aux[NewInput, Input]
    ): Layer.Aux[NewInput, Batch.Aux[OutputData, OutputDelta]] = {
      Compose(layer, toInput(toLayer(g)))
    }

    /**
      * Return a [[Layer]] that accepts input and will only forward.
      * If you want to test the accuracy of network assertions, you can not let your network backward, then you need to use `predict`.
      * @example{{{
      * import com.thoughtworks.deeplearning.DifferentiableAny._
      * def composeNetwork(implicit input: Symbolic[INDArray]##T) =???
      * val predictor=composeNetwork
      * predictor.predict(testData)
      * }}}
      */
    def predict[InputData, InputDelta](inputData: InputData)(
        implicit ev: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
          Batch.Aux[InputData, InputDelta],
          Batch.Aux[OutputData, OutputDelta]]
    ): OutputData = {
      managed(layer.forward(Literal[InputData](inputData))).acquireAndGet(_.value)
    }

    /**
      * Return a [[Layer]] that accepts input and will forward & backward.
      * If you want to train your network,you need your network backward, then you need to use `train`.
      * @example{{{
      * import com.thoughtworks.deeplearning.DifferentiableAny._
      * def composeNetwork(implicit input: Symbolic[INDArray]##T) =???
      * val yourNetwork=composeNetwork
      * yourNetwork.train(testData)
      * }}}
      */
    def train[InputData, InputDelta](inputData: InputData)(
        implicit ev: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] <:< Layer.Aux[
          Batch.Aux[InputData, InputDelta],
          Batch.Aux[OutputData, OutputDelta]],
        outputDataIsOutputDelta: Trainable[OutputData, OutputDelta]
    ): OutputData = {
      val outputBatch = layer.forward(Literal[InputData](inputData))
      try {
        val loss = outputBatch.value
        outputBatch.backward(outputDataIsOutputDelta(loss))
        loss
      } finally {
        outputBatch.close()
      }

    }

    /**
      * In DeepLearning.Scala,operation is not immediately run,
      * but first filled with placeholders, the entire network will be running ,then the real data will come into networks.
      * So if you want to see some vars's intermediate state,you need to use `withOutputDataHook`.
      * @example{{{
      * import com.thoughtworks.deeplearning.DifferentiableAny._
      * (var:From[INDArray]##T).withOutputDataHook{ data => println(data) }
      * }}}
      */
    def withOutputDataHook(hook: OutputData => Unit): Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = {
      WithOutputDataHook(layer, hook)
    }
  }

  /**
    * A helper that contains common boilerplate code for all layers.
    * @example{{{
    * import com.thoughtworks.deeplearning.DifferentiableAny._
    * }}}
    */
  implicit def toAnyLayerOps[A, Input <: Batch, OutputData, OutputDelta](a: A)(
      implicit toLayer: ToLayer.Aux[A, Input, OutputData, OutputDelta])
    : AnyLayerOps[Input, OutputData, OutputDelta] = {
    new AnyLayerOps(toLayer(a))
  }

  type ExistentialNothing = T forSome { type T >: Nothing <: Nothing }

  implicit def liftAny: ToLiteral.Aux[Any, Any, ExistentialNothing] = ToLiteral.fromData

  trait Trainable[-Data, +Delta] {
    def apply(data: Data): Delta
  }

}
