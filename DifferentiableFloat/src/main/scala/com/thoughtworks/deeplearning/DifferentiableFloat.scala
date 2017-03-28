package com.thoughtworks.deeplearning

import cats.Monoid
import cats.implicits._
import com.qifun.statelessFuture.Future
import com.thoughtworks.deeplearning.CumulativeTape.MonoidTape
import com.thoughtworks.deeplearning.DifferentiableFloat.Optimizers.Optimizer
import com.thoughtworks.deeplearning.Layer.Tape
import org.typelevel.future.sde.future
import org.typelevel.future.sde.future.AutoImports._
import shapeless.the

/**
  * A namespace of common operators for Float layers.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object DifferentiableFloat {

  private[deeplearning] trait FloatMonoidTape extends MonoidTape {

    override type Data = Float

    override type Delta = Float

    protected final def monoid: Monoid[Float] = implicitly[Monoid[Delta]]
  }

  /**
    * Optimizers of Float.
    *
    * @example{{{
    * implicit val optimizerFactory = new DifferentiableFloat.OptimizerFactory {
    *   override def FloatOptimizer(weight: Weight): Optimizer = {
    *     new LearningRate with L2Regularization {
    *
    *       var learningRate = 0.00003
    *
    *       override protected def l2Regularization: Float = 0.003
    *
    *       override protected def currentLearningRate(): Float = {
    *       learningRate * 0.75
    *       learningRate
    *      }
    *    }
    *  }
    * }
    * }}}
    */
  object Optimizers {

    trait Optimizer {
      def currentDelta(oldValue: Float, delta: Float): Float = delta

      final def updateFloat(oldValue: Float, delta: Float): Float = {
        oldValue - currentDelta(oldValue, delta)
      }
    }

    trait LearningRate extends Optimizer {

      protected def currentLearningRate(): Float

      override def currentDelta(oldValue: Float, delta: Float): Float = delta * currentLearningRate()
    }

    trait L1Regularization extends Optimizer {
      protected def l1Regularization: Float

      override def currentDelta(oldValue: Float, delta: Float): Float = {
        super.currentDelta(oldValue, delta + math.signum(oldValue) * l1Regularization)
      }
    }

    trait L2Regularization extends Optimizer {
      protected def l2Regularization: Float

      override def currentDelta(oldValue: Float, delta: Float): Float = {
        super.currentDelta(oldValue, delta + oldValue * l2Regularization)
      }
    }

  }

  object OptimizerFactory {
    implicit def shared(implicit optimizer: Optimizer): OptimizerFactory = new OptimizerFactory {
      override def floatOptimizer(weight: Weight): Optimizer = optimizer
    }
  }

  trait OptimizerFactory {
    def floatOptimizer(weight: Weight): Optimizer
  }

  object Weight {

    def apply(value: Float)(implicit optimizerFactory: OptimizerFactory) = new Weight(value) {
      override protected val optimizer: Optimizer = optimizerFactory.floatOptimizer(this)
    }

  }

  abstract case class Weight(var value: Float) extends Layer with Tape {

    override type Data = Float
    override type Delta = Float

    override type Input = Tape
    override type Output = Tape.Aux[Data, Delta]

    override final def isTrainable = true

    protected def optimizer: Optimizer

    override final def forward(any: Input) = Future {
      this
    }

    override final def duplicate(): Weight = this

    override def backward(delta: Delta): Future[Unit] = Future {
      synchronized {
        value = optimizer.updateFloat(value, delta)
      }
    }

    override final def close(): Future[Unit] = Future {}
  }

  object Tapes {
    abstract case class Plus(upstream0: Tape.Aux[Float, Float], upstream1: Tape.Aux[Float, Float])
        extends FloatMonoidTape {

      override val value: Float = upstream0.value + upstream1.value

      protected def upstream0Delta(outputDelta: Delta): Future[upstream0.Delta] = Future { outputDelta }

      protected def upstream1Delta(outputDelta: Delta): Future[upstream1.Delta] = Future { outputDelta }

    }

    abstract case class Negative(upstream0: Tape.Aux[Float, Float]) extends FloatMonoidTape {

      override protected def flush(delta: Float): Future[Unit] = Future {
        upstream0.backward(-delta)
      }

      override final def value: Float = -upstream0.value
    }

    abstract case class Reciprocal(upstream0: Tape.Aux[Float, Float]) extends FloatMonoidTape {

      override protected def flush(delta: Float): Future[Unit] = Future {
        val a = upstream0.value
        upstream0.backward(-delta / (a * a))
      }

      override final def value: Float = the[Numeric[Float]].one / upstream0.value
    }

    abstract case class Substract(upstream0: Tape.Aux[Float, Float], upstream1: Tape.Aux[Float, Float])
        extends FloatMonoidTape {

      override protected def flush(delta: Float): Future[Unit] = Future {
        upstream0.backward(delta)
        upstream1.backward(-delta)
      }

      override final def value: Float = upstream0.value - upstream1.value
    }

    abstract case class Times(upstream0: Tape.Aux[Float, Float], upstream1: Tape.Aux[Float, Float])
        extends FloatMonoidTape {

      protected def upstream0Delta(outputDelta: Delta): Future[upstream0.Delta] = Future {
        outputDelta * upstream1.value
      }

      protected def upstream1Delta(outputDelta: Delta): Future[upstream1.Delta] = Future {
        outputDelta * upstream0.value
      }

      override final def value: Float = upstream0.value * upstream1.value
    }

    abstract case class Log(upstream0: Tape.Aux[Float, Float]) extends FloatMonoidTape {

      override protected def flush(delta: Float): Future[Unit] = Future {
        upstream0.backward(delta / upstream0.value)
      }

      override final def value: Float = math.log(upstream0.value).toFloat
    }

    abstract case class Exp(upstream0: Tape.Aux[Float, Float]) extends FloatMonoidTape {

      override protected def flush(delta: Float): Future[Unit] = Future {
        upstream0.backward(delta * value)
      }

      override final def value: Float = math.exp(upstream0.value).toFloat
    }
  }

  import Tapes._

  implicit final class FloatTapeOps(floatTape: Tape.Aux[Float, Float]) {
    def +(right: Tape.Aux[Float, Float]): Plus = {
      CumulativeTape[Plus](floatTape.duplicate(), right.duplicate())
    }

    def - : Negative = {
      CumulativeTape[Negative](floatTape.duplicate())
    }

    def *(right: Tape.Aux[Float, Float]): Times = {
      CumulativeTape[Times](floatTape.duplicate(), right.duplicate())
    }

    def /(right: Tape.Aux[Float, Float]): Future[Times] = future {
      val r = CumulativeTape[Reciprocal](right.duplicate())
      try {
        CumulativeTape[Times](floatTape.duplicate(), r)
      } finally {
        r.close().!
      }
    }

    def log: Log = {
      CumulativeTape[Log](floatTape.duplicate())
    }

    def exp: Exp = {
      CumulativeTape[Exp](floatTape.duplicate())
    }

  }

  // implicit helpers, ops, ...
}
