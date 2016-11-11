package com.thoughtworks.deepLearning

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import shapeless.PolyDefns._
import shapeless.{Lazy, Poly1, Poly2}

//import com.thoughtworks.deepLearning.ToAst.Ast
import shapeless.DepFn1
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning._
import com.thoughtworks.deepLearning.any.ast.Identity
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

import scala.language.existentials

private[deepLearning] sealed trait ToAstLowPriorityImplicits {

  implicit def toAstOfType[Input0 <: Differentiable, OutputType <: DifferentiableType[_, _]]
    : ToAst.OfType[Ast[Input0, OutputType#Batch], Input0, OutputType] = {
    ToAst
      .astToAst[Input0, OutputType#Data, OutputType#Delta]
      .asInstanceOf[ToAst.OfType[Ast[Input0, OutputType#Batch], Input0, OutputType]]
  }

}

object ToAst extends ToAstLowPriorityImplicits {
  trait AstPoly1 extends Poly1 {
    implicit def toAstCase[Operand, Input <: Differentiable, OperandData, OperandDelta](
        implicit toAst: ToAst.Aux[Operand, Input, OperandData, OperandDelta],
        astCase: Lazy[Case[Ast[Input, Batch[OperandData, OperandDelta]]]]
    ): Case.Aux[Operand, astCase.value.Result] = {
      at { operand =>
        astCase.value(toAst(operand))
      }
    }
  }

  trait AstPoly2 extends Poly2 {
    implicit def toAstCase[LeftOperand,
                           RightOperand,
                           Input <: Differentiable,
                           LeftData,
                           LeftDelta,
                           RightData,
                           RightDelta](
        implicit leftToAst: ToAst.Aux[LeftOperand, Input, LeftData, LeftDelta],
        rightToAst: ToAst.Aux[RightOperand, Input, RightData, RightDelta],
        astCase: Lazy[Case[Ast[Input, Batch[LeftData, LeftDelta]], Ast[Input, Batch[RightData, RightDelta]]]]
    ): Case.Aux[LeftOperand, RightOperand, astCase.value.Result] = {
      at { (left, right) =>
        val leftAst = leftToAst(left)
        val rightAst = rightToAst(right)
        astCase.value(leftAst, rightAst)
      }
    }
  }

  type Aux[From, Input <: Differentiable, OutputData0, OutputDelta0] = ToAst[From, Input] {
    type OutputData = OutputData0
    type OutputDelta = OutputDelta0
  }

  type OfBatch[From, Input <: Differentiable, Output0 <: Differentiable] = ToAst[From, Input] {
    type Output = Output0
  }

  type OfType[From, Input <: Differentiable, Type <: DifferentiableType[_, _]] =
    ToAst.Aux[From, Input, differentiableType.Data, differentiableType.Delta] forSome { val differentiableType: Type }

  // FIXME: I don't know if invariance is required, please remove this line if Ast is enough
  //  type Ast[Input <: Differentiable, Output <: Differentiable] = Ast[Input, Output]

  implicit def astToAst[Input <: Differentiable, OutputData0, OutputDelta0]
    : ToAst.Aux[Ast[Input, Batch[OutputData0, OutputDelta0]], Input, OutputData0, OutputDelta0] =
    new ToAst[Ast[Input, Batch[OutputData0, OutputDelta0]], Input] {
      override type OutputData = OutputData0
      override type OutputDelta = OutputDelta0

      override def apply(ast: Ast[Input, Batch[OutputData, OutputDelta]]) = ast
    }

  implicit def inputTypeToAst[InputData, InputDelta]
    : ToAst.Aux[DifferentiableType[InputData, InputDelta], Batch[InputData, InputDelta], InputData, InputDelta] =
    new ToAst[DifferentiableType[InputData, InputDelta], Batch[InputData, InputDelta]] {
      override type OutputData = InputData
      override type OutputDelta = InputDelta

      override def apply(input: DifferentiableType[InputData, InputDelta]) =
        Identity[Batch[InputData, InputDelta]]()
    }
}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait ToAst[From, Input <: Differentiable] extends DepFn1[From] {
  type OutputData
  type OutputDelta
  type Output = Batch[OutputData, OutputDelta]
  type Out = Ast[Input, Output]
}
