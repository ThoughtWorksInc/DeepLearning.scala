package com.thoughtworks.deeplearning
import com.thoughtworks.raii.RAIITask

import scalaz.concurrent.Task

object TapeTask {

  def predict[OutputData, OutputDelta](forward: RAIITask[_ <: Tape.Aux[OutputData, OutputDelta]]): Task[OutputData] = {
    new Task(forward.map(_.data).run.run)
  }

  def train[OutputData, OutputDelta](forward: RAIITask[_ <: Tape.Aux[OutputData, OutputDelta]])(
      implicit trainable: Trainable[OutputData, OutputDelta]): Task[OutputData] = {
    val c: RAIITask[OutputData] = forward.flatMap[OutputData] { outputTape =>
      RAIITask.unmanaged(outputTape.backward(trainable(outputTape.data))).map { _ =>
        outputTape.data
      }
    }
    new Task(c.run.run)
  }

  trait Trainable[-Data, +Delta] {
    def apply(data: Data): RAIITask[_ <: Delta]
  }

}
