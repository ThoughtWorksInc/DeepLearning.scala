//package com.thoughtworks.deepLearning
//
//import cats._
//import cats.implicits._
//import com.thoughtworks.deepLearning.Differentiable.Batch
//import com.thoughtworks.deepLearning.DifferentiableFunction.{Ast, ToAst}
//import org.nd4s.Implicits._
//import com.thoughtworks.deepLearning._
//import com.thoughtworks.deepLearning.any.ast.Identity
//import org.nd4j.linalg.api.ndarray.INDArray
//import org.nd4j.linalg.factory.Nd4j
//import org.nd4j.linalg.ops.transforms.Transforms
//import shapeless.DepFn1
//import shapeless.DepFn2
//import shapeless.Lazy
//
///**
//  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
//  */
//object Poly {
//
//  trait Poly1 {
//
//    trait   Case[Input <: Differentiable, OperandData, OperandDelta]
//        extends DepFn1[Ast[Input, Batch[OperandData, OperandDelta]]]
//
//    final def apply[Input <: Differentiable, Operand, OperandData, OperandDelta, Out0](operand: Operand)(
//        implicit toAst: ToAst[Operand, Input, OperandData, OperandDelta],
//        polyCase: Case[Input, OperandData, OperandDelta] {
//          type Out = Out0
//        }): Out0 = {
//      polyCase(toAst(operand))
//    }
//  }
//
//  trait Poly2 {
//
//    object Case {
//      type Aux[Input <: Differentiable, LeftOperandData, LeftOperandDelta, RightOperandData, RightOperandDelta, Out0] =
//        Case[Input, LeftOperandData, LeftOperandDelta, RightOperandData, RightOperandDelta] {
//          type Out = Out0
//        }
//    }
//
//    trait Case[Input <: Differentiable, LeftOperandData, LeftOperandDelta, RightOperandData, RightOperandDelta]
//        extends DepFn2[Ast[Input, Batch[LeftOperandData, LeftOperandDelta]],
//                       Ast[Input, Batch[RightOperandData, RightOperandDelta]]]
//
//    final def apply[Input <: Differentiable : Identity,
//                    LeftOperand,
//                    LeftOperandData,
//                    LeftOperandDelta,
//                    RightOperand,
//                    RightOperandData,
//                    RightOperandDelta,
//                    Out0](leftOperand: LeftOperand, rightOperand: RightOperand)(
//        implicit leftToAst: ToAst[LeftOperand, Input, LeftOperandData, LeftOperandDelta],
//        rightToAst: ToAst[RightOperand, Input, RightOperandData, RightOperandDelta],
//        polyCase: Case[Input, LeftOperandData, LeftOperandDelta, RightOperandData, RightOperandDelta] {
//          type Out = Out0
//        }) = {
//      polyCase(leftToAst(leftOperand), rightToAst(rightOperand))
//    }
//
//  }
//
//}
