package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Conversion._
import shapeless.{Lazy, Poly1, Poly2}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Poly {

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

  object MathMethods {
    object - extends LayerPoly2
    object + extends LayerPoly2
    object * extends LayerPoly2
    object / extends LayerPoly2
  }

  implicit final class MathOps[Left](left: Left) {

    def -[Right](right: Right)(implicit methodCase: MathMethods.-.Case[Left, Right]): methodCase.Result =
      MathMethods.-(left, right)

    def +[Right](right: Right)(implicit methodCase: MathMethods.+.Case[Left, Right]): methodCase.Result =
      MathMethods.+(left, right)

    def *[Right](right: Right)(implicit methodCase: MathMethods.*.Case[Left, Right]): methodCase.Result =
      MathMethods.*(left, right)

    def /[Right](right: Right)(implicit methodCase: MathMethods./.Case[Left, Right]): methodCase.Result =
      MathMethods./(left, right)

  }

  object MathFunctions {

    object log extends LayerPoly1
    object exp extends LayerPoly1
    object abs extends LayerPoly1
    object max extends LayerPoly2
    object min extends LayerPoly2

  }

}
