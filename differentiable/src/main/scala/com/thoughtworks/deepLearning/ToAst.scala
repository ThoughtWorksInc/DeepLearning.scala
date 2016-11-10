package com.thoughtworks.deepLearning

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.ToAst.InvariantAst
import shapeless.DepFn1
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning._
import com.thoughtworks.deepLearning.any.ast.Identity
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

import scala.language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait ToAst[From, Input <: Differentiable] extends DepFn1[From] {
  type OutputData
  type OutputDelta
  type Output = Batch[OutputData, OutputDelta]
  type Out = InvariantAst[Input, Output]
}

object ToAst {

  type Aux[From, Input <: Differentiable, OutputData0, OutputDelta0] = ToAst[From, Input] {
    type OutputData = OutputData0
    type OutputDelta = OutputDelta0
  }

  type OfType[From, Input <: Differentiable, Type <: DifferentiableType] = ToAst[From, Input] {
    type OutputData = differentiableType.Data
    type OutputDelta = differentiableType.Delta
  } forSome { val differentiableType: Type }

  // FIXME: I don't know if invariance is required, please remove this line if Ast is enough
  type InvariantAst[Input <: Differentiable, Output <: Differentiable] = Ast[Input, Output]

  implicit def astToAst[Input <: Differentiable, OutputData0, OutputDelta0]
    : ToAst.Aux[Ast[Input, Batch[OutputData0, OutputDelta0]], Input, OutputData0, OutputDelta0] =
    new ToAst[Ast[Input, Batch[OutputData0, OutputDelta0]], Input] {
      override type OutputData = OutputData0
      override type OutputDelta = OutputDelta0

      override def apply(ast: Ast[Input, Batch[OutputData, OutputDelta]]) = ast
    }

  implicit def inputTypeToAst[InputType <: DifferentiableType]
    : ToAst.Aux[InputType, Batch[InputType#Data, InputType#Delta], InputType#Data, InputType#Delta] =
    new ToAst[InputType, Batch[InputType#Data, InputType#Delta]] {
      override type OutputData = InputType#Data
      override type OutputDelta = InputType#Delta
      override def apply(input: InputType) =
        Identity[Batch[InputType#Data, InputType#Delta]]()
    }
}
