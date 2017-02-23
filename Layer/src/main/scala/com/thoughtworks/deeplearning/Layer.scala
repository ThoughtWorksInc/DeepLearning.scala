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

trait Layer {

  import Layer._

  type Input <: Batch

  type Output <: Batch

  def forward(input: Input): Future.Stateless[Output]

}
