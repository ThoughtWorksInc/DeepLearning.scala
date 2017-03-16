package com.thoughtworks.deeplearning

import java.io.Closeable

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait CheckedCloseable extends Closeable {

  protected def forceClose(): Unit

  private var isReleased = false

  override final def close(): Unit = {
    val wasRelease = synchronized {
      val wasRelease = isReleased
      if (!wasRelease) {
        isReleased = true
      }
      wasRelease
    }
    if (!wasRelease) {
      forceClose()
    }
  }

}

object CheckedCloseable {

  trait FinallyCloseable extends Closeable {
    override protected final def finalize(): Unit = close()
  }

}
