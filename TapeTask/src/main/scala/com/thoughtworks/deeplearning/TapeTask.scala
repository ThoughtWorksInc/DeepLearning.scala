package com.thoughtworks.deeplearning
import com.thoughtworks.deeplearning.Tape.{Aux, Literal}
import com.thoughtworks.raii.{RAIIFuture, RAIITask}

import scalaz.{EitherT, \/}
import scalaz.concurrent.{Future, Task}

object TapeTask {

  def predict[OutputData, OutputDelta](forward: RAIITask[_ <: Tape.Aux[OutputData, OutputDelta]]): Task[OutputData] = {
    new Task(forward.map(_.data).run.run)
  }

  def train[OutputData, OutputDelta](forward: RAIITask[_ <: Tape.Aux[OutputData, OutputDelta]])(
      implicit outputDataIsOutputDelta: Trainable[OutputData, OutputDelta]): Task[OutputData] = {
    val c: EitherT[RAIIFuture, Throwable, OutputData] =
      forward
        .flatMap[OutputData] { outputTape =>
          val loss = outputTape.data
          RAIITask.unmanaged(outputTape.backward(outputDataIsOutputDelta(loss))).map { _: Unit =>
            loss
          }
        }
    new Task(c.run.run)
  }

  trait Trainable[-Data, +Delta] {
    def apply(data: Data): Future[Delta]
  }

}
