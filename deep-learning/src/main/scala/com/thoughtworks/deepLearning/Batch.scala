package com.thoughtworks.deepLearning

import scala.annotation.elidable

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object Batch {

  /** @template */
  type Aux[+Data0, -Delta0] = Batch {
    type Data <: Data0
    type Delta >: Delta0
  }

  trait Unshared extends Batch {

    private[Unshared] final class ClosingFlag {
      var closed = false
      @elidable(elidable.ASSERTION)
      def assertNotClosed() = {
        assert(!closed)
        closed = true
      }
    }

    @elidable(elidable.ASSERTION)
    private val closingFlag = new ClosingFlag

    protected def closeUpstreams(): Unit

    override final def close() = {
      closingFlag.assertNotClosed()
      closeUpstreams()
    }
  }

}

trait Batch extends AutoCloseable {
  type Data
  type Delta

  def backward(delta: Delta): Unit

  def value: Data
}
