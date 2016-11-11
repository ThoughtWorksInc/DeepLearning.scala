package com.thoughtworks
import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import shapeless.{Poly1, Poly2}

import scala.language.implicitConversions
//
//import cats._
//import cats.implicits._
//import com.thoughtworks.deepLearning.Poly.{Poly1, Poly2}
//import org.nd4s.Implicits._
//import org.nd4j.linalg.api.ndarray.INDArray
//import org.nd4j.linalg.factory.Nd4j
//import org.nd4j.linalg.ops.transforms.Transforms
//
///**
//  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
//  */
package object deepLearning {
//
  object log extends Poly1
//  object exp extends Poly1
//  object abs extends Poly1
  object max extends Poly2

  object - extends Poly2
  object + extends Poly2
  object * extends Poly2
  object / extends Poly2

  implicit final class AstOps[Left](left: Left) {

    def -[Right](right: Right)(implicit cse: deepLearning.-.Case[Left, Right]): cse.Result =
      deepLearning.-(left, right)

    def +[Right](right: Right)(implicit cse: deepLearning.+.Case[Left, Right]): cse.Result =
      deepLearning.+(left, right)

    def *[Right](right: Right)(implicit cse: deepLearning.*.Case[Left, Right]): cse.Result =
      deepLearning.*(left, right)

    def /[Right](right: Right)(implicit cse: deepLearning./.Case[Left, Right]): cse.Result =
      deepLearning./(left, right)

  }

  implicit def autoToLiteral[A, Input <: Differentiable, OutputData, OutputDelta](a: A)(
      implicit toAst: ToAst.Aux[A, Input, OutputData, OutputDelta]): Ast[Input, Batch[OutputData, OutputDelta]] = {
    toAst(a)
  }

  implicit final class ToLiteral[A](a: A) {
    def toLiteral[Input <: Differentiable, OutputData, OutputDelta](
        implicit toAst: ToAst.Aux[A, Input, OutputData, OutputDelta]): Ast[Input, Batch[OutputData, OutputDelta]] = {
      toAst(a)
    }
  }

}
