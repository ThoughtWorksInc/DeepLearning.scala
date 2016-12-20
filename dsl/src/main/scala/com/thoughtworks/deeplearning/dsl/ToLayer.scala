package com.thoughtworks.deeplearning
package dsl

import com.thoughtworks.deeplearning.Layer.Batch
import shapeless._

import language.existentials

// TODO: Move to dsl library
private[deeplearning] sealed trait ToLayerLowPriorityImplicits {

  implicit def toLayerOfType[Input0 <: Batch, OutputType <: BackPropagationType[_, _]]
    : ToLayer.OfType[Layer.Aux[Input0, OutputType#Batch], Input0, OutputType] = {
    ToLayer
      .layerToLayer[Input0, BackPropagationType.DataOf[OutputType], BackPropagationType.DeltaOf[OutputType]]
      .asInstanceOf[ToLayer.OfType[Layer.Aux[Input0, OutputType#Batch], Input0, OutputType]]
  }

}

object ToLayer extends ToLayerLowPriorityImplicits {
  trait LayerPoly1 extends Poly1 {
    implicit def toLayerCase[Operand, Input <: Batch, OperandData, OperandDelta](
        implicit toLayer: ToLayer.Aux[Operand, Input, OperandData, OperandDelta],
        layerCase: Lazy[Case[Layer.Aux[Input, Batch.Aux[OperandData, OperandDelta]]]]
    ): Case.Aux[Operand, layerCase.value.Result] = {
      at { operand =>
        layerCase.value(toLayer(operand))
      }
    }
  }

  trait LayerPoly2 extends Poly2 {
    implicit def toLayerCase[LeftOperand, RightOperand, Input <: Batch, LeftData, LeftDelta, RightData, RightDelta](
        implicit leftToLayer: ToLayer.Aux[LeftOperand, Input, LeftData, LeftDelta],
        rightToLayer: ToLayer.Aux[RightOperand, Input, RightData, RightDelta],
        layerCase: Lazy[
          Case[Layer.Aux[Input, Batch.Aux[LeftData, LeftDelta]], Layer.Aux[Input, Batch.Aux[RightData, RightDelta]]]]
    ): Case.Aux[LeftOperand, RightOperand, layerCase.value.Result] = {
      at { (left, right) =>
        val leftLayer = leftToLayer(left)
        val rightLayer = rightToLayer(right)
        layerCase.value(leftLayer, rightLayer)
      }
    }
  }

  type Aux[From, Input <: Batch, OutputData0, OutputDelta0] = ToLayer[From, Input] {
    type OutputData = OutputData0
    type OutputDelta = OutputDelta0
  }

  type OfType[From, Input <: Batch, OutputType <: BackPropagationType[_, _]] =
    ToLayer.Aux[From, Input, differentiableType.Data, differentiableType.Delta] forSome {
      val differentiableType: OutputType
    }

  implicit def layerToLayer[Input <: Batch, OutputData0, OutputDelta0]
    : ToLayer.Aux[Layer.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]], Input, OutputData0, OutputDelta0] =
    new ToLayer[Layer.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]], Input] {
      override type OutputData = OutputData0
      override type OutputDelta = OutputDelta0

      override def apply(layer: Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = layer
    }

  implicit def literalToLayer[From, InputData, InputDelta, OutputData0, OutputDelta0](
      implicit inputType: BackPropagationType[InputData, InputDelta],
      toLiteral: ToLiteral.Aux[From, OutputData0, OutputDelta0])
    : ToLayer.Aux[From, Batch.Aux[InputData, InputDelta], OutputData0, OutputDelta0] = {
    new ToLayer[From, Batch.Aux[InputData, InputDelta]] {
      override type OutputData = OutputData0
      override type OutputDelta = OutputDelta0
      override def apply(from: From) = toLiteral(from)
    }
  }

}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait ToLayer[From, Input <: Batch] extends DepFn1[From] {
  type OutputData
  type OutputDelta
  type Output = Batch.Aux[OutputData, OutputDelta]
  type Out = Layer.Aux[Input, Output]
}
