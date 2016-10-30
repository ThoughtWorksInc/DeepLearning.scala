package com.thoughtworks.deepLearning.dsl

import cats.Eval
import com.thoughtworks.deepLearning.NeuralNetwork
import com.thoughtworks.deepLearning.NeuralNetwork.Batch

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
sealed trait Coproduct extends Any {
  type Data <: shapeless.Coproduct
  type Delta <: shapeless.Coproduct
}

sealed trait CNil extends Coproduct {
  type Data = shapeless.CNil
  type Delta = shapeless.CNil
}

sealed trait :+:[Head <: Any, Tail <: Coproduct] extends Coproduct {
  type Data = shapeless.:+:[Head#Data, Tail#Data]
  type Delta = shapeless.:+:[Head#Delta, Tail#Delta]
}

object Coproduct {

  final case class CConsHead[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
  TailDelta <: shapeless.Coproduct](
      ccons: NeuralNetwork.Aux[Input0,
                               Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ) extends NeuralNetwork {

    final class Output private[CConsHead] (
        upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
        extends Batch {
      override type Data = HeadData
      override type Delta = HeadDelta
      type Input >: Input0

      val value =
        upstream.value.asInstanceOf[shapeless.Inl[HeadData, TailData]].head

      override def backward(delta: Delta): Unit = {
        upstream.backward(shapeless.Inl(delta))
      }

      override def close(): Unit = {
        upstream.close()
      }

    }

    type Input = Input0

    override def forward(input: Input): Output = {
      new Output(ccons.forward(input))
    }

  }

  final case class CConsTail[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
  TailDelta <: shapeless.Coproduct](
      ccons: NeuralNetwork.Aux[Input0,
                               Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ) extends NeuralNetwork {

    final class Output private[CConsTail] (
        upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
        extends Batch {
      override type Data = TailData
      override type Delta = TailDelta
      type Input >: Input0

      val value =
        upstream.value.asInstanceOf[shapeless.Inr[TailData, TailData]].tail

      override def backward(delta: Delta): Unit = {
        upstream.backward(shapeless.Inr(delta))
      }

      override def close(): Unit = {
        upstream.close()
      }
    }

    type Input = Input0

    override def forward(input: Input): Output = {
      new Output(ccons.forward(input))
    }

  }

  final case class IsInl[Input0 <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
  TailDelta <: shapeless.Coproduct](
      ccons: NeuralNetwork.Aux[Input0,
                               Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]
  ) extends NeuralNetwork {

    final class Output private[IsInl] (
        upstream: Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]])
        extends Boolean
        with Batch {
      type Input >: Input0
      val value = upstream.value match {
        case shapeless.Inl(_) => Eval.now(true)
        case shapeless.Inr(_) => Eval.now(false)
      }

      override def backward(delta: Eval[scala.Boolean]): Unit = {}

      override def close(): Unit = {
        upstream.close()
      }
    }

    type Input = Input0

    override def forward(input: Input): Output = {
      new Output(ccons.forward(input))
    }
  }

  final case class DifferentiableInr[Input0 <: Batch, TailData <: shapeless.Coproduct,
  TailDelta <: shapeless.Coproduct](tail: NeuralNetwork.Aux[Input0, Batch.Aux[TailData, TailDelta]])
      extends NeuralNetwork {

    type Input = Input0

    final class Output private[DifferentiableInr] (tailBatch: Batch.Aux[TailData, TailDelta]) extends Batch {
      def value = shapeless.Inr(tailBatch.value: TailData)

      type Data = shapeless.Inr[Nothing, TailData]
      type Delta = shapeless.:+:[scala.Any, TailDelta]

      override def backward(delta: shapeless.:+:[scala.Any, TailDelta]): Unit = {
        delta match {
          case shapeless.Inr(tailDelta) => tailBatch.backward(tailDelta)
          case shapeless.Inl(_) =>
        }
      }

      override def close(): Unit = {
        tailBatch.close()
      }
    }

    override def forward(input: Input0): Output = {
      new Output(tail.forward(input))
    }

  }

  final case class DifferentiableInl[Input0 <: Batch, HeadData, HeadDelta](
      head: NeuralNetwork.Aux[Input0, Batch.Aux[HeadData, HeadDelta]])
      extends NeuralNetwork {

    type Input = Input0

    final class Output private[DifferentiableInl] (headBatch: Batch.Aux[HeadData, HeadDelta]) extends Batch {
      def value = shapeless.Inl(headBatch.value: HeadData)

      type Data = shapeless.Inl[HeadData, Nothing]
      type Delta = shapeless.:+:[HeadDelta, shapeless.Coproduct]

      override def backward(delta: shapeless.:+:[HeadDelta, shapeless.Coproduct]): Unit = {
        delta match {
          case shapeless.Inl(headDelta) => headBatch.backward(headDelta)
          case shapeless.Inr(_) =>
        }
      }

      override def close(): Unit = {
        headBatch.close()
      }
    }

    override def forward(input: Input0): Output = {
      new Output(head.forward(input))
    }

  }

}
