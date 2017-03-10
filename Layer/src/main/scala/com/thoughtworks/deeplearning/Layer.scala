package com.thoughtworks.deeplearning

import com.qifun.statelessFuture.Future

import language.existentials
import language.implicitConversions
import language.higherKinds
import scala.annotation.elidable

object Layer {

  private[deeplearning] trait CloseableOnce extends AutoCloseable {

    private[CloseableOnce] final class ClosingFlag {
      var closed = false
      @elidable(elidable.ASSERTION)
      def close() = {
        assert(!closed)
        closed = true
      }

      @elidable(elidable.ASSERTION)
      def assertClosed() = {
        assert(closed)
      }
    }

    // FIXME: @elidable should only be used for def
    @elidable(elidable.ASSERTION)
    private val closingFlag = new ClosingFlag

    override def close() = {
      closingFlag.close()
    }

    override protected def finalize(): Unit = {
      closingFlag.assertClosed()
    }
  }

  object Batch {

    /** @template */
    type Aux[+Data0, -Delta0] = Batch {
      type Data <: Data0
      type Delta >: Delta0
    }

  }

  /**
    * 在DeepLearning.Scala中，每个输入和输出都是一个Batch，即每个输入和输出都包含Data和Delta，
    */
  trait Batch extends AutoCloseable {
    type Data
    type Delta

    // TODO: rename to `duplicate`?
    /**
      * Returns a new [[Batch]] that shares the same [[value]] and [[backward]] behavior with this [[Batch]].
      * @note The newly created [[Batch]] and this [[Batch]] must be [[close]]d independently.
      */
    def addReference(): Batch.Aux[Data, Delta]

    protected def forceBackward(delta: Delta): Future.Stateless[Unit]

    def isTrainable: Boolean

    @inline
    final def backward(delta: Delta): Future.Stateless[Unit] = {
      if (isTrainable) {
        forceBackward(delta)
      } else {
        Future(())
      }
    }

    def value: Data
  }

  /** @template */
  type Aux[-Input0 <: Batch, +Output0 <: Batch] =
    Layer {
      type Input >: Input0
      type Output <: Output0
    }

}

/**
  * The layer of DeepLearning.Scala is similar to the layer of neural networks ,But there are some details is difference
  * The layer in the neural networks means a layer of network, but the layer in DeepLearning.Scala means a operation,
  * many operations could also compose together to be one layer(at this time , layer is means layer of neural networks).
  * Every operation in DeepLearning.Scala is a layer. Every layer contains input,output and forward.
  */
trait Layer {

  import Layer._

  type Input <: Batch

  type Output <: Batch

  def forward(input: Input): Future.Stateless[Output]

}
