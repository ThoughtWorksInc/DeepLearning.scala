package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Conversion._
import shapeless.PolyDefns.~>
import shapeless.{DepFn2, Lazy, Poly1, Poly2}

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
    implicit def toLayerCase[LeftOperand,
                             RightOperand,
                             Input <: Batch,
                             LeftData,
                             LeftDelta,
                             RightData,
                             RightDelta,
                             Result](
        implicit leftToLayer: ToLayer.Aux[LeftOperand, Input, LeftData, LeftDelta],
        rightToLayer: ToLayer.Aux[RightOperand, Input, RightData, RightDelta],
        layerCase: Case.Aux[Layer.Aux[Input, Batch.Aux[LeftData, LeftDelta]],
                        Layer.Aux[Input, Batch.Aux[RightData, RightDelta]],
                        Result]
    ): Case.Aux[LeftOperand, RightOperand, Result] = {
      at { (left, right) =>
        val leftLayer = leftToLayer(left)
        val rightLayer = rightToLayer(right)
        layerCase(leftLayer, rightLayer)
      }
    }
  }

  trait ToLayerPoly2 {
    def at[Operand1, Operand2] =
      new ~>[({ type T[x] = (Operand1, Operand2) => x })#T, ({ type T[x] = Case.Aux[Operand1, Operand2, x] })#T] {
        override def apply[T](f: (Operand1, Operand2) => T): ToLayerPoly2.this.Case.Aux[Operand1, Operand2, T] =
          new ToLayerPoly2.this.Case[Operand1, Operand2] {
            override type Out = T
            override def apply(t: Operand1, u: Operand2) = f(t, u)
          }
      }
    object Case {
      type Aux[Operand1, Operand2, Out0] = Case[Operand1, Operand2] {
        type Out = Out0
      }
    }
    trait Case[Operand1, Operand2] extends DepFn2[Operand1, Operand2]
    def apply[Operand1, Operand2, Input <: Batch, LeftData, LeftDelta, RightData, RightDelta, Out0](
        operand1: Operand1,
        operand2: Operand2)(
        implicit leftToLayer: ToLayer.Aux[Operand1, Input, LeftData, LeftDelta],
        rightToLayer: ToLayer.Aux[Operand2, Input, RightData, RightDelta],
        methodCase: Case.Aux[Layer.Aux[Input, Batch.Aux[LeftData, LeftDelta]],
                             Layer.Aux[Input, Batch.Aux[RightData, RightDelta]],
                             Out0]
    ): Out0 = {
      methodCase(leftToLayer(operand1), rightToLayer(operand2))
    }
  }

  object MathMethods {
    object - extends ToLayerPoly2
    object + extends LayerPoly2
    object * extends LayerPoly2
    object / extends LayerPoly2
  }

  implicit final class MathOps[Left](left: Left) {

    def -[Operand2, Input <: Batch, LeftData, LeftDelta, RightData, RightDelta, Out0](right: Operand2)(
        implicit leftToLayer: ToLayer.Aux[Left, Input, LeftData, LeftDelta],
        rightToLayer: ToLayer.Aux[Operand2, Input, RightData, RightDelta],
        methodCase: MathMethods.-.Case.Aux[Layer.Aux[Input, Batch.Aux[LeftData, LeftDelta]],
                                           Layer.Aux[Input, Batch.Aux[RightData, RightDelta]],
                                           Out0]
    ): Out0 =
      MathMethods.-(left, right)(leftToLayer, rightToLayer, methodCase)

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
