package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Tape
import cats.implicits._
import cats._

import com.qifun.statelessFuture.Future

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait CumulativeTape extends Tape {

  /**
    * Returns a wrapped [[com.thoughtworks.deeplearning.Layer.Tape Tape]] able to detect error of closing more than once if ASSERTION is enabled,
    * or returns this [[CumulativeTape]] itself when ASSERTION is disabled hence no check.
    */
  override final def duplicate(): this.type = {
    val newCount = synchronized {
      val newCount = count + 1
      count = newCount
      newCount
    }
    assert(newCount >= 1)
    this
  }

  private[deeplearning] var count: Int = 1

  protected def flush(): Unit

  protected def closeUpstreams(): Unit

  override final def close(): Unit = {
    val newCount = synchronized {
      val newCount = count - 1
      count = newCount
      newCount
    }
    assert(newCount >= 0)
    if (newCount == 0) {
      flush()
      closeUpstreams()
    }
  }
}
object CumulativeTape {

  trait MonoidTape extends CumulativeTape {

    private var deltaAccumulator: Delta = monoid.empty

    /**
      * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
      */
    protected def rawBackward(delta: Delta): Unit

    implicit protected def monoid: Monoid[Delta]

    override protected final def flush(): Unit = {
      rawBackward(synchronized {
        val delta = deltaAccumulator
        deltaAccumulator = monoid.empty
        delta
      })
    }

    override final def forceBackward(outputDelta: Delta) = {
      synchronized {
        deltaAccumulator = deltaAccumulator |+| outputDelta
      }
      Future(())
    }
  }

  trait SemigroupTape extends CumulativeTape {

    private var deltaAccumulator: Option[Delta] = None

    /**
      * Performs the underlying backward pass with all `outputDelta`s that previously received from [[forceBackward]].
      */
    protected def rawBackward(delta: Delta): Unit

    implicit protected def semigroup: Semigroup[Delta]

    override protected final def flush(): Unit = {
      synchronized {
        val delta = deltaAccumulator
        deltaAccumulator = None
        delta
      }.foreach(rawBackward)
    }

    override final def forceBackward(outputDelta: Delta) = {
      synchronized {
        deltaAccumulator |+|= Some(outputDelta)
      }
      Future(())
    }

  }

}
