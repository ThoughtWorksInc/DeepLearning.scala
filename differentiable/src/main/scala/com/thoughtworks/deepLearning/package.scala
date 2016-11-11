package com.thoughtworks
import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.ToAst.{AstPoly1, AstPoly2}

import scala.language.implicitConversions

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object deepLearning {

  implicit final class AstOps[Left](left: Left) {

    def -[Right](right: Right)(implicit methodCase: AstMethods.-.Case[Left, Right]): methodCase.Result =
      AstMethods.-(left, right)

    def +[Right](right: Right)(implicit methodCase: AstMethods.+.Case[Left, Right]): methodCase.Result =
      AstMethods.+(left, right)

    def *[Right](right: Right)(implicit methodCase: AstMethods.*.Case[Left, Right]): methodCase.Result =
      AstMethods.*(left, right)

    def /[Right](right: Right)(implicit methodCase: AstMethods./.Case[Left, Right]): methodCase.Result =
      AstMethods./(left, right)

  }

  implicit def autoToLiteral[A, Input <: Differentiable, OutputData, OutputDelta](a: A)(
      implicit toAst: ToAst.Aux[A, Input, OutputData, OutputDelta]): Ast[Input, Batch[OutputData, OutputDelta]] = {
    toAst(a)
  }

  implicit final class ToLiteralOps[A](a: A) {
    def toLiteral[Input <: Differentiable, OutputData, OutputDelta](
        implicit toAst: ToAst.Aux[A, Input, OutputData, OutputDelta]): Ast[Input, Batch[OutputData, OutputDelta]] = {
      toAst(a)
    }
  }

  object log extends AstPoly1
  object exp extends AstPoly1
  object abs extends AstPoly1
  object max extends AstPoly2

}

package deepLearning {

  object AstMethods {
    object - extends AstPoly2
    object + extends AstPoly2
    object * extends AstPoly2
    object / extends AstPoly2
  }

}
