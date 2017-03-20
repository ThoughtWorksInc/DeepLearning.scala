package com.thoughtworks.deeplearning

import java.io.Closeable

/**
  * A [[Closeable]] that tracks the if it is closed.
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait IsClosed extends Closeable {

  protected def forceClose(): Unit

  private var _isClosed = false

  final def isClosed = _isClosed

  /**
    * Calls [[forceClose]] and then marks this [[IsClosed]] as closed if this [[IsClosed]] was not closed; does nothing otherwise.
    */
  override final def close(): Unit = {
    val wasClosed = synchronized {
      val wasClosed = _isClosed
      if (!wasClosed) {
        _isClosed = true
      }
      wasClosed
    }
    if (!wasClosed) {
      forceClose()
    }
  }

}

object IsClosed {

  trait FinallyCloseable extends Closeable {
    override protected final def finalize(): Unit = close()
  }

}
