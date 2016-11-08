package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.any.Any
import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.Batch._

import scala.language.implicitConversions
import cats.Eval
import com.thoughtworks.deepLearning.any.ast.{Identity, Literal}
import com.thoughtworks.deepLearning.boolean.ast.If
import com.thoughtworks.deepLearning.double.ast._
import com.thoughtworks.deepLearning.boolean.utilities.Boolean

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object double {

  /** @template */
  type Double = utilities.Double

  implicit final class DoubleOps[Input <: Batch](differentiable: WidenAst[Input, Double#Widen]) {
    def +[RightInput <: Input](right: WidenAst[RightInput, Double#Widen]): WidenAst[RightInput, Double#Widen] = {
      Add(differentiable, right)
    }
    def -[RightInput <: Input](right: WidenAst[RightInput, Double#Widen]): WidenAst[RightInput, Double#Widen] = {
      Add(differentiable, Negative(right))
    }
    def /[RightInput <: Input](right: WidenAst[RightInput, Double#Widen]): WidenAst[RightInput, Double#Widen] = {
      Multiply(differentiable, Reciprocal(right))
    }
    def *[RightInput <: Input](right: WidenAst[RightInput, Double#Widen]): WidenAst[RightInput, Double#Widen] = {
      Multiply(differentiable, right)
    }
    def <[RightInput <: Input](right: WidenAst[RightInput, Double#Widen]): WidenAst[RightInput, Boolean#Widen] = {
      LessThan(differentiable, right)
    }
    def unary_- : WidenAst[Input, Double#Widen] = {
      Negative(differentiable)
    }
  }

  implicit def nativeDoubleToDoubleOps(nativeDouble: scala.Double): DoubleOps[Batch] = {
    DoubleOps(Literal(Eval.now(nativeDouble)))
  }

  implicit def doubleLiteral[Input <: Batch: Identity](nativeDouble: scala.Double): WidenAst[Input, Double#Widen] = {
    Literal(Eval.now(nativeDouble))
  }

  class InputTypePair[Data, Delta]

  implicit final class NativeDoubleOps(nativeDouble: scala.Double) {
    def toLiteral[Input <: Batch: Identity] = doubleLiteral(nativeDouble)
    def toWeight[Input <: Batch: Identity](implicit learningRate: LearningRate): WidenAst[Input, Double#Widen] = {
      Weight(nativeDouble)
    }
  }

  def exp[Input <: Batch](doubleAst: WidenAst[Input, Double#Widen]): WidenAst[Input, Double#Widen] = {
    Exp(doubleAst)
  }

  def log[Input <: Batch](doubleAst: WidenAst[Input, Double#Widen]): WidenAst[Input, Double#Widen] = {
    Log(doubleAst)
  }

  def abs[Input <: Batch](doubleAst: WidenAst[Input, Double#Widen]): WidenAst[Input, Double#Widen] = {
    If(doubleAst < 0.0, -doubleAst, doubleAst)
  }

}
