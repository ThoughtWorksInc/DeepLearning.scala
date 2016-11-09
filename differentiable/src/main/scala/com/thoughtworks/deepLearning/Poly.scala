package com.thoughtworks.deepLearning

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.DifferentiableFunction.{Ast, ToAst}
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning._
import com.thoughtworks.deepLearning.any.ast.Identity
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import shapeless.DepFn1
import shapeless.DepFn2
import shapeless.Lazy

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Poly {

  trait Poly1 {

    trait Case[Input <: Differentiable, OperandData, OperandDelta]
        extends DepFn1[Ast[Input, Batch[OperandData, OperandDelta]]]

    final def apply[Input <: Differentiable, Operand, OperandData, OperandDelta, Out0](operand: Operand)(
        implicit toAst: ToAst[Operand, Input, OperandData, OperandDelta],
        polyCase: Case[Input, OperandData, OperandDelta] {
          type Out = Out0
        }): Out0 = {
      polyCase(toAst(operand))
    }
  }

  trait Poly2 {

    trait Case[Input <: Differentiable, LeftOperandData, LeftOperandDelta, RightOperandData, RightOperandDelta]
        extends DepFn2[Ast[Input, Batch[LeftOperandData, LeftOperandDelta]],
                       Ast[Input, Batch[RightOperandData, RightOperandDelta]]]
    //
    // trait Case {
    //   type Input <: Differentiable
    //
    //   type LeftOperandData
    //   type LeftOperandDelta
    //   type RightOperandData
    //   type RightOperandDelta
    //   type OutputData
    //   type OutputDelta
    //   protected[Poly2] def apply(leftOperand: Ast[Input, Batch[LeftOperandData, LeftOperandDelta]],
    //                              rightOperand: Ast[Input, Batch[RightOperandData, RightOperandDelta]])
    //     : Ast[Input, Batch[OutputData, OutputDelta]]
    // }
    //
    // object Case {
    //   type Aux[Input0 <: Differentiable,
    //            LeftOperandData0,
    //            LeftOperandDelta0,
    //            RightOperandData0,
    //            RightOperandDelta0,
    //            OutputData0,
    //            OutputDelta0] = Case {
    //     type Input = Input0
    //     type LeftOperandData = LeftOperandData0
    //     type LeftOperandDelta = LeftOperandDelta0
    //     type RightOperandData = RightOperandData0
    //     type RightOperandDelta = RightOperandDelta0
    //     type OutputData = OutputData0
    //     type OutputDelta = OutputDelta0
    //   }
    // }

    final def apply[Input <: Differentiable,
                    LeftOperand,
                    LeftOperandData,
                    LeftOperandDelta,
                    RightOperand,
                    RightOperandData,
                    RightOperandDelta,
                    Out0](leftOperand: LeftOperand, rightOperand: RightOperand)(
        implicit leftToAst: ToAst[LeftOperand, Input, LeftOperandData, LeftOperandDelta],
        rightToAst: ToAst[RightOperand, Input, RightOperandData, RightOperandDelta],
        polyCase: Case[Input, LeftOperandData, LeftOperandDelta, RightOperandData, RightOperandDelta] {
          type Out = Out0
        }) = {
      polyCase(leftToAst(leftOperand), rightToAst(rightOperand))
    }

  }

}
