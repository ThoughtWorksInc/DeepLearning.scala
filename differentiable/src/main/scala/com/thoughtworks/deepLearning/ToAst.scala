package com.thoughtworks.deepLearning

import cats._
import cats.implicits._
import com.thoughtworks.deepLearning.Differentiable.Batch
import com.thoughtworks.deepLearning.DifferentiableFunction.Ast
import com.thoughtworks.deepLearning.ToAst.{DataOf, DeltaOf}
import shapeless.Lazy

//import com.thoughtworks.deepLearning.ToAst.Ast
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
  type Out = Ast[Input, Output]
}

private[deepLearning] sealed trait ToAstLowPriorityImplicits {

  implicit def toAstOfType[Input0 <: Differentiable, OutputType <: DifferentiableType[_, _]]
    : ToAst.OfType[Ast[Input0, OutputType#Batch], Input0, OutputType] = {
    ToAst
      .astToAst[Input0, OutputType#Data, OutputType#Delta]
      .asInstanceOf[ToAst.OfType[Ast[Input0, OutputType#Batch], Input0, OutputType]]
  }

}

object ToAst extends ToAstLowPriorityImplicits {

  type Aux[From, Input <: Differentiable, OutputData0, OutputDelta0] = ToAst[From, Input] {
    type OutputData = OutputData0
    type OutputDelta = OutputDelta0
  }

  type OfBatch[From, Input <: Differentiable, Output0 <: Differentiable] = ToAst[From, Input] {
    type Output = Output0
  }

  type DataOf[Type <: DifferentiableType[_, _]] = (differentiableType.Data forSome { val differentiableType: Type })
  type DeltaOf[Type <: DifferentiableType[_, _]] = (differentiableType.Delta forSome { val differentiableType: Type })

  type OfType[From, Input <: Differentiable, Type <: DifferentiableType[_, _]] =
    ToAst.Aux[From, Input, DataOf[Type], DeltaOf[Type]]

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
