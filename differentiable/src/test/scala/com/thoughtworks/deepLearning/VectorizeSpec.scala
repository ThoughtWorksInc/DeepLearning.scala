package com.thoughtworks.deepLearning

import cats.Eval
import cats.implicits._
import com.thoughtworks.deepLearning.Differentiable.DifferentiableHCons.{Head, Tail}
import com.thoughtworks.deepLearning.Differentiable._
import org.nd4j.linalg.api.ndarray.INDArray
import org.scalatest._
import scala.language.implicitConversions
import scala.language.existentials

final case class DifferentiableOps[InputType <: DifferentiableType.Any]() {

  type InputData = InputType#Data
  type InputDelta = InputType#Delta

  type Input = Batch.Aux[InputData, InputDelta]
  val input = Id[InputData, InputDelta]()

  type Any = DifferentiableOps.Any[Input]

  type Boolean = DifferentiableOps.Boolean[Input]

  implicit def toBoolean(
      differentiable: Differentiable.Aux[Input, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]]) =
    new Boolean(differentiable)

  type Double = DifferentiableOps.Double[Input]

  implicit def toDouble(differentiable: Differentiable.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) =
    new Double(differentiable)

  type Coproduct = DifferentiableOps.Coproduct[Input]
  type CNil = DifferentiableOps.CNil[Input]

  type :+:[Head <: Any, Tail <: Coproduct] =
    DifferentiableOps.CCons[Input, Head#OutputData, Head#OutputDelta, Tail#OutputData, Tail#OutputDelta]

  implicit def toCCons[HeadData, HeadDelta, TailData <: shapeless.Coproduct, TailDelta <: shapeless.Coproduct](
      differentiable: Differentiable.Aux[
        Input,
        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) =
    new DifferentiableOps.CCons(differentiable)

  type HList = DifferentiableOps.HList[Input]
  type HNil = DifferentiableOps.HNil[Input]
  type ::[Head <: Any, Tail <: HList] =
    DifferentiableOps.HCons[Input, Head#OutputData, Head#OutputDelta, Tail#OutputData, Tail#OutputDelta]

  implicit def toHCons[HeadData, HeadDelta, TailData <: shapeless.HList, TailDelta <: shapeless.Coproduct](
      differentiable: Differentiable.Aux[
        Input,
        Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) =
    new DifferentiableOps.HCons(differentiable)

  def `throw`(throwable: => Throwable) = {
    Throw(Eval.later(throwable))
  }

  // TODO: upcast for HCons and CCons
}

object DifferentiableOps {

  // TODO: Let Input contravariant
  trait Any[Input <: Batch] extends scala.Any {
    type OutputData
    type OutputDelta
    type Network = Differentiable.Aux[Input, Batch.Aux[OutputData, OutputDelta]]

    val differentiable: Network

  }

  object Any {

    private type Aux[Input <: Batch, OutputData0, OutputDelta0] = Any[Input] {
      type OutputData = OutputData0
      type OutputDelta = OutputDelta0
    }

    implicit def getDifferentiable[Input <: Batch, OutputData0, OutputDelta0](
        any: Any.Aux[Input, OutputData0, OutputDelta0]) =
      any.differentiable

  }

  final class Boolean[Input <: Batch](
      val differentiable: Differentiable.Aux[Input, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]])
      extends AnyVal
      with Any[Input] {
    override type OutputData = Eval[scala.Boolean]
    override type OutputDelta = Eval[scala.Boolean]
  }

  final class Double[Input <: Batch](
      val differentiable: Differentiable.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]])
      extends AnyVal
      with Any[Input] {
    override type OutputData = Eval[scala.Double]
    override type OutputDelta = Eval[scala.Double]
  }

  sealed trait Coproduct[Input <: Batch] extends scala.Any with Any[Input] {
    override type OutputData <: shapeless.Coproduct
    override type OutputDelta <: shapeless.Coproduct
  }

  final class CCons[Input <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
  TailDelta <: shapeless.Coproduct](
      val differentiable: Differentiable.Aux[
        Input,
        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]])
      extends Coproduct[Input] {
    override type OutputData = shapeless.:+:[HeadData, TailData]
    override type OutputDelta = shapeless.:+:[HeadDelta, TailDelta]

    private lazy val head = CConsHead(differentiable)
    private lazy val tail = CConsTail(differentiable)

    def choice[ResultData, ResultDelta](
        caseHead: Differentiable.Aux[Input, Batch.Aux[HeadData, HeadDelta]] => Differentiable.Aux[
          Input,
          Batch.Aux[ResultData, ResultDelta]])(
        caseTail: Differentiable.Aux[Input, Batch.Aux[TailData, TailDelta]] => Differentiable.Aux[
          Input,
          Batch.Aux[ResultData, ResultDelta]]): Differentiable.Aux[Input, Batch.Aux[ResultData, ResultDelta]] = {
      If(IsInl(differentiable), caseHead(head), caseTail(tail))
    }

  }

  object CCons {
    implicit def covariant[
        Input <: Batch,
        HeadData,
        HeadDelta,
        TailData <: shapeless.Coproduct,
        TailDelta <: shapeless.Coproduct,
        HeadData1 >: HeadData,
        HeadDelta1 <: HeadDelta,
        TailData1 >: TailData <: shapeless.Coproduct,
        TailDelta1 <: TailDelta
    ](ccons: CCons[Input, HeadData, HeadDelta, TailData, TailDelta])
      : CCons[Input, HeadData1, HeadDelta1, TailData1, TailDelta1] = {
      new CCons[Input, HeadData1, HeadDelta1, TailData1, TailDelta1](ccons.differentiable)
    }
  }

  sealed trait CNil[Input <: Batch] extends scala.Any with Coproduct[Input] {
    override type OutputData = shapeless.CNil
    override type OutputDelta = shapeless.CNil
  }

  sealed trait HList[Input <: Batch] extends scala.Any with Any[Input] {
    override type OutputData <: shapeless.HList
    override type OutputDelta <: shapeless.Coproduct
  }

  final class HCons[Input <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList, TailDelta <: shapeless.Coproduct](
      val differentiable: Differentiable.Aux[
        Input,
        Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]])
      extends AnyVal
      with HList[Input] {
    override type OutputData = shapeless.::[HeadData, TailData]
    override type OutputDelta = shapeless.:+:[HeadDelta, TailDelta]

    def head = Head(differentiable)

    def tail = Tail(differentiable)
  }

  final class HNil[Input <: Batch](
      val differentiable: Differentiable.Aux[Input, Batch.Aux[shapeless.HNil, shapeless.CNil]])
      extends AnyVal
      with HList[Input] {
    override type OutputData = shapeless.HNil
    override type OutputDelta = shapeless.CNil
  }

}

object DifferentiableType {

  trait Any {
    type Data
    type Delta
  }

  trait Double extends Any {
    type Data = Eval[scala.Double]
    type Delta = Eval[scala.Double]
  }

  trait Array2D extends Any {
    type Data = Eval[scala.Double]
    type Delta = Eval[scala.Double]
  }

  trait HList extends Any {
    type Data <: shapeless.HList
    type Delta <: shapeless.Coproduct
  }

  trait HNil extends HList {
    type Data = shapeless.HNil
    type Delta = shapeless.CNil
  }

  trait ::[Head <: Any, Tail <: HList] extends HList {
    type Data = shapeless.::[Head#Data, Tail#Data]
    type Delta = shapeless.:+:[Head#Delta, Tail#Delta]
  }

  trait Coproduct extends Any {
    type Data <: shapeless.Coproduct
    type Delta <: shapeless.Coproduct
  }

  trait CNil extends Coproduct {
    type Data = shapeless.CNil
    type Delta = shapeless.CNil
  }

  trait :+:[Head <: Any, Tail <: Coproduct] extends Coproduct {
    type Data = shapeless.:+:[Head#Data, Tail#Data]
    type Delta = shapeless.:+:[Head#Delta, Tail#Delta]
  }

}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class VectorizeSpec extends FreeSpec with Matchers {

  import VectorizeSpec._

  "Convert HMatrix to Array2D" in {
    /*
     TODO: 最终目标是生成一个预测神经网络和一个训练神经网络
     为了生成这两个网络，需要生成若干处理Array2D的全连接层、InputData到Array2D的转换、Array2D到Row的转换、Array2D到Double loss的转换

     InputData到Array2D的转换可以从InputData到若干Double的转换做起

     目前可以暂时使用HList而不是直接用case class的神经网络，将来可以直接使用case class

     */

    implicit val learningRate = new LearningRate {
      override def apply(): Double = 0.0003
    }

    val factory = DifferentiableOps[InputType]()
    import factory._

    val toArray2D: Differentiable.Aux[Batch.Aux[InputType#Data, InputType#Delta], Batch.Aux[Eval[INDArray], Eval[INDArray]]] = {

      val field0 = input.head
      val rest0 = input.tail
      val field1 = rest0.head
      val rest1 = rest0.tail
      val field2 = rest1.head
      val rest2 = rest1.tail
      val field3 = rest2.head
      val rest3 = rest2.tail

      val field0Flag0: Double#Network = field0.choice { _: HNil#Network =>
        Literal(Eval.now(0.0))
      } { _ =>
        Literal(Eval.now(1.0))
      }

      val field0Flag1: Double#Network = field0.choice { unknown: HNil#Network =>
        DoubleWeight(0.5)
      } {
        _.choice { knownField0 =>
          knownField0.choice { unset: HNil#Network =>
            Literal(Eval.now(0.0))
          } { someValue: (Double :+: CNil)#Network =>
            Literal(Eval.now(1.0))
          }
        } { _: CNil#Network =>
          `throw`(new IllegalArgumentException)
        }
      }

      val field0Value: Double#Network = field0.choice { unknown: HNil#Network =>
        DoubleWeight(0.5)
      } {
        _.choice { knownField0 =>
          knownField0.choice { unset: HNil#Network =>
            DoubleWeight(0.5)
          } {
            _.choice[Eval[scala.Double], Eval[scala.Double]] { value: Double#Network =>
              value
            } { _: CNil#Network =>
              `throw`(new IllegalArgumentException): Double#Network
            }
          }
        } { _: CNil#Network =>
          `throw`(new IllegalArgumentException)
        }

      }

      //      val isField1Unknown: Boolean = IsInl(field1)
      //      val defaultValueForField1Value0 = DoubleWeight(0.5)
      //      val field1Choice0 = CConsHead(CConsTail(field1))
      //      val isField1Value0: Boolean = IsInl(field1Choice0)
      //      //      val isField1Value1 = IsInl(field1Choice1)
      //
      //      val field1Flag0 = {
      //        If(
      //          isField1Unknown,
      //          Literal(Eval.now(0.0)),
      //          Literal(Eval.now(1.0))
      //        )
      //      }
      //      val field1Value0 = {
      //        val defaultValue = DoubleWeight(0.5)
      //        If(
      //          isField1Unknown,
      //          defaultValue,
      //          If(
      //            isField1Value0,
      //            Literal(Eval.now(0.0)),
      //            Literal(Eval.now(1.0))
      //          )
      //        )
      //      }
      //
      //      val isField3Unknown: Boolean = IsInl(field3)
      //      val defaultValueForField3Value0 = DoubleWeight(0.5)
      //      val field3Choice0 = CConsHead(CConsTail(field3))
      //      val field3Choice1 = CConsTail(field3Choice0)
      //      val isField3Value0: Boolean = IsInl(field3Choice0)
      //      val isField3Value1: Boolean = IsInl(field3Choice1)
      //
      //      val field3Flag0 = {
      //        If(
      //          isField3Unknown,
      //          Literal(Eval.now(0.0)),
      //          Literal(Eval.now(1.0))
      //        )
      //      }
      //      val field3Value0 = {
      //        val defaultValue = DoubleWeight(0.5)
      //        If(
      //          isField3Unknown,
      //          defaultValue,
      //          If(
      //            isField3Value0,
      //            Literal(Eval.now(0.0)),
      //            Literal(Eval.now(1.0))
      //          )
      //        )
      //      }
      //      val field3Value1 = {
      //        val defaultValue = DoubleWeight(0.5)
      //        If(
      //          isField3Unknown,
      //          defaultValue,
      //          If(
      //            isField3Value1,
      //            Literal(Eval.now(0.0)),
      //            Literal(Eval.now(1.0))
      //          )
      //        )
      //      }
      //
      ???

    }

    //    val predict: Differentiable.Aux[Batch.Aux[InputData, _], Batch.Aux[Row, _]] = ???
    //
    //    val train: Differentiable.Aux[Batch.Aux[InputData :: ExpectedLabelData :: HNil, _], Batch.Aux[Eval[Double], _]] = ???

  }

}

object VectorizeSpec {

//  import shapeless._
  import DifferentiableType._
  type Nullable[A <: Any] = HNil :+: A :+: CNil

//  // TODO: 将来要重构，把类型映射的元信息提到某个trait上。就不用写一遍Data再写一遍Delta了。

  type InputField[A <: Any] = HNil :+: A :+: CNil

  type LabelField[A <: Any] = HNil :+: A :+: CNil

  type Enum0 = HNil :+: HNil :+: CNil
  type Enum1 = HNil :+: HNil :+: HNil :+: CNil

  type Row = Nullable[Double] :: Enum0 :: Double :: Enum1 :: HNil

  type InputType =
    InputField[Nullable[Double]] :: InputField[Enum0] :: InputField[Double] :: InputField[Enum1] :: HNil

  type ExpectedLabel =
    LabelField[Nullable[Double]] :: LabelField[Enum0] :: LabelField[Double] :: LabelField[Enum1] :: HNil

}
