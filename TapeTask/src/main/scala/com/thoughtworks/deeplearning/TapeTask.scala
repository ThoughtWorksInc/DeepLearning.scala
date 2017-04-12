package com.thoughtworks.deeplearning
import com.thoughtworks.deeplearning.Tape.{Aux, Literal}
import com.thoughtworks.raii.ResourceFactoryT.ResourceT
import com.thoughtworks.raii.{RAIIFuture, RAIITask, ResourceFactoryT}

import scalaz.{EitherT, \/, \/-}
import scalaz.concurrent.{Future, Task}

object TapeTask {

  def predict[OutputData, OutputDelta](forward: RAIITask[_ <: Tape.Aux[OutputData, OutputDelta]]): Task[OutputData] = {
    new Task(forward.map(_.data).run.run)
  }

  def train[OutputData, OutputDelta](forward: RAIITask[_ <: Tape.Aux[OutputData, OutputDelta]])(
      implicit trainable: Trainable[OutputData, OutputDelta]): Task[OutputData] = {
    val c: RAIITask[OutputData] = forward.flatMap[OutputData] { outputTape =>
      trainable(outputTape.data).flatMap { delta: OutputDelta =>
        RAIITask.unmanaged(outputTape.backward(Future.now(delta))).map { _: Unit =>
          outputTape.data
        }
      }
    }
    new Task(c.run.run)
  }

  trait Trainable[-Data, +Delta] {
    def apply(data: Data): RAIITask[_ <: Delta]
  }

}
