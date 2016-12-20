package com.thoughtworks.deeplearning.utilities

import annotation.elidable

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

  @elidable(elidable.ASSERTION)
  private val closingFlag = new ClosingFlag

  override def close() = {
    closingFlag.close()
  }

  override protected def finalize(): Unit = {
    closingFlag.assertClosed()
  }
}
