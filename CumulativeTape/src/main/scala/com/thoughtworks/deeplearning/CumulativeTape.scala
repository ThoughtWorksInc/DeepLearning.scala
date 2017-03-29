package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.Layer.Tape
import cats.implicits._
import cats._
import com.thoughtworks.future.Continuation.Task
import com.thoughtworks.future.sde.task
import task.AwaitOps

import scala.util.control.TailCalls._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait CumulativeTape {

  protected def flush(): Task[Unit]

  protected def releaseUpstreams(): Task[Unit]
  protected def retainUpstreams(): Task[Unit]

}

object CumulativeTape {

  trait Trainable extends Tape.Trainable with CumulativeTape {

    /**
      * Returns a wrapped [[com.thoughtworks.deeplearning.Layer.Tape Tape]] able to detect error of closing more than once if ASSERTION is enabled,
      * or returns this [[CumulativeTape]] itself when ASSERTION is disabled hence no check.
      */
    final def retain(): TailRec[Unit] = {
      val newCount = synchronized {
        val newCount = count + 1
        count = newCount
        newCount
      }
      assert(newCount >= 1)
      done(())
    }

    private[deeplearning] var count: Int = 1

    final def release(): Task[Unit] = task {
      val newCount = synchronized {
        val newCount = count - 1
        count = newCount
        newCount
      }
      assert(newCount >= 0)
      if (newCount == 0) {
        flush().!
        releaseUpstreams().!
      }
    }
  }

  trait MonoidTape extends CumulativeTape { this: Tape =>

    private var deltaAccumulator: Delta = monoid.empty

    implicit protected def monoid: Monoid[Delta]

    /**
      * Performs the underlying backward pass with all `outputDelta`s that previously received from [[backward]].
      */
    protected def flush(deltaSum: Delta): Task[Unit]

    override protected def flush(): Task[Unit] = {
      flush(synchronized {
        val delta = deltaAccumulator
        deltaAccumulator = monoid.empty
        delta
      })
    }

    def backward(outputDelta: Delta): Task[Unit] = {
      task {
        synchronized {
          deltaAccumulator = deltaAccumulator |+| outputDelta
        }
      }
    }
  }

  trait SemigroupTape extends CumulativeTape { this: Tape =>

    private var deltaAccumulator: Option[Delta] = None

    implicit protected def semigroup: Semigroup[Delta]

    /**
      * Performs the underlying backward pass with all `outputDelta`s that previously received from [[backward]].
      */
    protected def flush(deltaSum: Delta): Task[Unit]

    override protected def flush(): Task[Unit] = {
      synchronized {
        val deltaOption = deltaAccumulator
        deltaAccumulator = None
        deltaOption
      } match {
        case None => task(())
        case Some(delta) => flush(delta)
      }
    }

    protected final def appendDelta(outputDelta: Delta): Task[Unit] = {
      task {
        synchronized {
          deltaAccumulator |+|= Some(outputDelta)
        }
      }
    }

  }
//
//  trait UnaryTape extends CumulativeTape {
//    val upstream0: Tape
//
//    override protected final def closeUpstreams(): Task[Unit] = {
//      upstream0.close()
//    }
//
//    protected def upstream0Delta(outputDelta: Delta): Task[upstream0.Delta]
//
//  }
//
//  object UnaryTape {
//
//    trait TrainableTape extends UnaryTape {
//      override final def isTrainable: Boolean = true
//    }
//
//    trait UntrainableTape extends UnaryTape {
//      override final def isTrainable: Boolean = false
//
//      override final protected def flush(): Task[Unit] = task(())
//
//      override final def backward(delta: Delta) = task(())
//    }
//
//  }
//
//  trait BinaryTape extends CumulativeTape {
//    val upstream0: Tape
//    val upstream1: Tape
//
//    override protected final def closeUpstreams(): Task[Unit] = task {
//      // TODO: parallelize the two release calls
//      upstream0.close().!
//      upstream1.close().!
//    }
//
//    protected def upstream0Delta(outputDelta: Delta): Task[upstream0.Delta]
//
//    protected def upstream1Delta(outputDelta: Delta): Task[upstream1.Delta]
//  }
//
//  object BinaryTape {
//
//    trait BothTrainableTape extends BinaryTape {
//      override final def isTrainable: Boolean = true
//      protected def flush(outputDelta: Delta): Task[Unit] = task {
//        upstream0.backward(upstream0Delta(outputDelta).!).!
//        upstream1.backward(upstream1Delta(outputDelta).!).!
//      }
//    }
//
//    trait Upstream0TrainableTape extends BinaryTape {
//      override final def isTrainable: Boolean = true
//      protected def flush(outputDelta: Delta): Task[Unit] = {
//        upstream0Delta(outputDelta).flatMap(upstream0.backward)
//      }
//    }
//
//    trait Upstream1TrainableTape extends BinaryTape {
//      protected def flush(outputDelta: Delta): Task[Unit] = {
//        upstream1Delta(outputDelta).flatMap(upstream1.backward)
//      }
//
//      override final def isTrainable: Boolean = true
//    }
//
//    trait BothUntrainableTape extends BinaryTape {
//      override final def isTrainable: Boolean = false
//
//      protected def flush(outputDelta: Delta): Task[Unit] = task(())
//
//      override final def backward(delta: Delta) = task(())
//    }
//
//  }
//
//  def apply[AbstractTape <: CumulativeTape](upstream0: Tape)(
//      implicit trainableConstructor: Constructor[(upstream0.type) => AbstractTape with UnaryTape.TrainableTape],
//      untrainableConstructor: Constructor[(upstream0.type) => AbstractTape with UnaryTape.UntrainableTape])
//    : AbstractTape = {
//    if (upstream0.isTrainable) {
//      trainableConstructor.newInstance(upstream0)
//    } else {
//      untrainableConstructor.newInstance(upstream0)
//    }
//  }
//
//  def apply[AbstractTape <: CumulativeTape](upstream0: Tape, upstream1: Tape)(
//      implicit bothTrainableConstructor: Constructor[
//        (upstream0.type, upstream1.type) => AbstractTape with BinaryTape.BothTrainableTape],
//      upstream0TrainableConstructor: Constructor[
//        (upstream0.type, upstream1.type) => AbstractTape with BinaryTape.Upstream0TrainableTape],
//      upstream1TrainableConstructor: Constructor[
//        (upstream0.type, upstream1.type) => AbstractTape with BinaryTape.Upstream1TrainableTape],
//      bothUntrainableConstructor: Constructor[
//        (upstream0.type, upstream1.type) => AbstractTape with BinaryTape.BothUntrainableTape]): AbstractTape = {
//    if (upstream0.isTrainable) {
//      if (upstream1.isTrainable) {
//        bothTrainableConstructor.newInstance(upstream0, upstream1)
//      } else {
//        upstream0TrainableConstructor.newInstance(upstream0, upstream1)
//
//      }
//    } else {
//      if (upstream1.isTrainable) {
//        upstream1TrainableConstructor.newInstance(upstream0, upstream1)
//      } else {
//        bothUntrainableConstructor.newInstance(upstream0, upstream1)
//      }
//    }
//  }

}
