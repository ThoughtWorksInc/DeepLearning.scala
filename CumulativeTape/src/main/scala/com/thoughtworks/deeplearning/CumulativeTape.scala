package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Tape
import cats.implicits._
import cats._
import com.qifun.statelessFuture.Future
import org.typelevel.future.sde.future
import org.typelevel.future.sde.future.AutoImports._

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

  protected def flush(): Future[Unit]

  protected def closeUpstreams(): Future[Unit]

  override final def close(): Future[Unit] = future {
    val newCount = synchronized {
      val newCount = count - 1
      count = newCount
      newCount
    }
    assert(newCount >= 0)
    if (newCount == 0) {
      flush().!
      closeUpstreams().!
    }
  }

}
object CumulativeTape {

  trait MonoidTape extends CumulativeTape {

    private var deltaAccumulator: Delta = monoid.empty

    implicit protected def monoid: Monoid[Delta]

    /**
      * Performs the underlying backward pass with all `outputDelta`s that previously received from [[backward]].
      */
    protected def flush(deltaSum: Delta): Future[Unit]

    override protected def flush(): Future[Unit] = {
      if (isTrainable) {
        flush(synchronized {
          val delta = deltaAccumulator
          deltaAccumulator = monoid.empty
          delta
        })
      } else {
        future(())
      }
    }

    def backward(outputDelta: Delta): Future[Unit] = {
      future {
        synchronized {
          deltaAccumulator = deltaAccumulator |+| outputDelta
        }
      }
    }
  }

  trait SemigroupTape extends CumulativeTape {

    private var deltaAccumulator: Option[Delta] = None

    implicit protected def semigroup: Semigroup[Delta]

    /**
      * Performs the underlying backward pass with all `outputDelta`s that previously received from [[backward]].
      */
    protected def flush(deltaSum: Delta): Future[Unit]

    override protected def flush(): Future[Unit] = {
      synchronized {
        val deltaOption = deltaAccumulator
        deltaAccumulator = None
        deltaOption
      } match {
        case None => future(())
        case Some(delta) => flush(delta)
      }
    }

    protected final def appendDelta(outputDelta: Delta): Future[Unit] = {
      future {
        synchronized {
          deltaAccumulator |+|= Some(outputDelta)
        }
      }
    }

  }

  trait UnaryTape extends CumulativeTape {
    val upstream0: Tape

    override protected final def closeUpstreams(): Future[Unit] = {
      upstream0.close()
    }

    protected def upstream0Delta(outputDelta: Delta): Future[upstream0.Delta]

  }

  object UnaryTape {

    trait TrainableTape extends UnaryTape {
      override final def isTrainable: Boolean = true
    }

    trait UntrainableTape extends UnaryTape {
      override final def isTrainable: Boolean = false

      override final protected def flush(): Future[Unit] = future(())

      override final def backward(delta: Delta) = future(())
    }

  }

  trait BinaryTape extends CumulativeTape {
    val upstream0: Tape
    val upstream1: Tape

    override protected final def closeUpstreams(): Future[Unit] = future {
      // TODO: parallelize the two close calls
      upstream0.close().!
      upstream1.close().!
    }

    protected def upstream0Delta(outputDelta: Delta): Future[upstream0.Delta]

    protected def upstream1Delta(outputDelta: Delta): Future[upstream1.Delta]
  }

  object BinaryTape {

    trait BothTrainableTape extends BinaryTape {
      override final def isTrainable: Boolean = true
      protected def flush(outputDelta: Delta): Future[Unit] = future {
        upstream0.backward(upstream0Delta(outputDelta).!).!
        upstream1.backward(upstream1Delta(outputDelta).!).!
      }
    }

    trait Upstream0TrainableTape extends BinaryTape {
      override final def isTrainable: Boolean = true
      protected def flush(outputDelta: Delta): Future[Unit] = {
        upstream0Delta(outputDelta).flatMap(upstream0.backward)
      }
    }

    trait Upstream1TrainableTape extends BinaryTape {
      protected def flush(outputDelta: Delta): Future[Unit] = {
        upstream1Delta(outputDelta).flatMap(upstream1.backward)
      }

      override final def isTrainable: Boolean = true
    }

    trait BothUntrainableTape extends BinaryTape {
      override final def isTrainable: Boolean = false

      protected def flush(outputDelta: Delta): Future[Unit] = future(())

      override final def backward(delta: Delta) = future(())
    }

  }

  def apply[AbstractTape <: CumulativeTape](upstream0: Tape)(
      implicit trainableConstructor: Constructor[(upstream0.type) => AbstractTape with UnaryTape.TrainableTape],
      untrainableConstructor: Constructor[(upstream0.type) => AbstractTape with UnaryTape.UntrainableTape])
    : AbstractTape = {
    if (upstream0.isTrainable) {
      trainableConstructor.newInstance(upstream0)
    } else {
      untrainableConstructor.newInstance(upstream0)
    }
  }

  def apply[AbstractTape <: CumulativeTape](upstream0: Tape, upstream1: Tape)(
      implicit bothTrainableConstructor: Constructor[
        (upstream0.type, upstream1.type) => AbstractTape with BinaryTape.BothTrainableTape],
      upstream0TrainableConstructor: Constructor[
        (upstream0.type, upstream1.type) => AbstractTape with BinaryTape.Upstream0TrainableTape],
      upstream1TrainableConstructor: Constructor[
        (upstream0.type, upstream1.type) => AbstractTape with BinaryTape.Upstream1TrainableTape],
      bothUntrainableConstructor: Constructor[
        (upstream0.type, upstream1.type) => AbstractTape with BinaryTape.BothUntrainableTape]): AbstractTape = {
    if (upstream0.isTrainable) {
      if (upstream1.isTrainable) {
        bothTrainableConstructor.newInstance(upstream0, upstream1)
      } else {
        upstream0TrainableConstructor.newInstance(upstream0, upstream1)

      }
    } else {
      if (upstream1.isTrainable) {
        upstream1TrainableConstructor.newInstance(upstream0, upstream1)
      } else {
        bothUntrainableConstructor.newInstance(upstream0, upstream1)
      }
    }
  }

}
