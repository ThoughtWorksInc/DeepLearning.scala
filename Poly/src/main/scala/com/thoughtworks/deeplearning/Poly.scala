package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Tape
import com.thoughtworks.deeplearning.Symbolic._
import shapeless.{Lazy, Poly1, Poly2}

/**
  * A namespace of common math operators.
  *
  * [[Poly.MathMethods MathMethods]] and [[Poly.MathFunctions MathFunctions]] provide functions like [[Poly.MathMethods.+ +]], [[Poly.MathMethods.- -]], [[Poly.MathMethods.* *]], [[Poly.MathMethods./ /]],
  * [[Poly.MathFunctions.log log]], [[Poly.MathFunctions.abs abs]], [[Poly.MathFunctions.max max]], [[Poly.MathFunctions.min min]] and [[Poly.MathFunctions.exp exp]], those functions been implements in specific Differentiable Object such as [[DifferentiableINDArray]]
  *
  * @see [[DifferentiableINDArray.Double+INDArray]]
  * @see [[DifferentiableINDArray.exp(INDArray)]]
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Poly {

  /**
    * A [[shapeless.Poly1 unary polymorphic function]] that accepts some kind of [[Layer]]s or values able to convert to those kind of layers.
    *
    * @see [[https://github.com/milessabin/shapeless/wiki/Feature-overview:-shapeless-2.0.0#polymorphic-function-values]]
    */
  trait LayerPoly1 extends Poly1 {
    implicit def toLayerCase[Operand, Input <: Tape, OperandData, OperandDelta](
        implicit toLayer: ToLayer.Aux[Operand, Input, OperandData, OperandDelta],
        layerCase: Lazy[Case[Layer.Aux[Input, Tape.Aux[OperandData, OperandDelta]]]]
    ): Case.Aux[Operand, layerCase.value.Result] = {
      at { operand =>
        layerCase.value(toLayer(operand))
      }
    }
  }

  /**
    * A [[shapeless.Poly2 binary polymorphic function]] that accepts some kind of [[Layer]]s or values able to convert to those kind of layers.
    *
    * @see [[https://github.com/milessabin/shapeless/wiki/Feature-overview:-shapeless-2.0.0#polymorphic-function-values]]
    */
  trait LayerPoly2 extends Poly2 {
    implicit def toLayerCase[LeftOperand, RightOperand, Input <: Tape, LeftData, LeftDelta, RightData, RightDelta](
        implicit leftToLayer: ToLayer.Aux[LeftOperand, Input, LeftData, LeftDelta],
        rightToLayer: ToLayer.Aux[RightOperand, Input, RightData, RightDelta],
        layerCase: Lazy[
          Case[Layer.Aux[Input, Tape.Aux[LeftData, LeftDelta]], Layer.Aux[Input, Tape.Aux[RightData, RightDelta]]]]
    ): Case.Aux[LeftOperand, RightOperand, layerCase.value.Result] = {
      at { (left, right) =>
        val leftLayer = leftToLayer(left)
        val rightLayer = rightToLayer(right)
        layerCase.value(leftLayer, rightLayer)
      }
    }
  }

  /**
    * Provide [[Poly.MathMethods.+ +]], [[Poly.MathMethods.- -]], [[Poly.MathMethods.* *]] and [[Poly.MathMethods./ /]] which is called in [[Poly.MathOps MathOps]].
    */
  object MathMethods {
    object - extends LayerPoly2
    object + extends LayerPoly2
    object * extends LayerPoly2
    object / extends LayerPoly2
  }

  /**
    * [[Poly.MathMethods.+ +]], [[Poly.MathMethods.- -]], [[Poly.MathMethods.* *]], [[Poly.MathMethods./ /]] are provide by [[MathMethods]] and implement in specific `DifferentiableType` such as [[DifferentiableINDArray]]
    */
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

  /**
    * Provide [[Poly.MathFunctions.log log]], [[Poly.MathFunctions.abs abs]], [[Poly.MathFunctions.max max]], [[Poly.MathFunctions.min min]] and [[Poly.MathFunctions.exp exp]]
    */
  object MathFunctions {

    object log extends LayerPoly1
    object exp extends LayerPoly1
    object abs extends LayerPoly1
    object max extends LayerPoly2
    object min extends LayerPoly2

  }

}
