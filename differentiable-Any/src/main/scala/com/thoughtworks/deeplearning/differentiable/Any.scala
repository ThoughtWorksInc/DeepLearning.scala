package com.thoughtworks.deeplearning.differentiable

import com.thoughtworks.deeplearning.Tape
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.covariant.ResourceT
import com.thoughtworks.raii.covariant.ResourceT._

import scalaz.concurrent.Future._
import scalaz.concurrent.Task
import scalaz.std.`try`.toDisjunction
import scalaz.syntax.all._

object Any extends AnyCompanion {

  def predict[OutputData, OutputDelta](forward: Do[Tape[OutputData, OutputDelta]]): Task[OutputData] = {
    val Do(doOutputData) = forward.map(_.data)
    new Task(ResourceT.run(ResourceT(doOutputData).map(toDisjunction)))
  }

  def train[OutputData, OutputDelta](forward: Do[Tape[OutputData, OutputDelta]])(
      implicit trainable: Trainable[OutputData, OutputDelta]): Task[OutputData] = {

    val doOutputData: Do[OutputData] = forward.flatMap[OutputData] { outputTape =>
      Do.delay(outputTape.backward(trainable(outputTape.data))).map { _ =>
        outputTape.data
      }
    }

    val Do(futureOutputData) = doOutputData
    new Task(ResourceT.run(ResourceT(futureOutputData).map(toDisjunction)))
  }
}

//workaround for https://github.com/scala/bug/issues/10306
private[deeplearning] abstract class AnyCompanion {
  trait Trainable[-Data, +Delta] {
    def apply(data: Data): Do[Delta]
  }
}
