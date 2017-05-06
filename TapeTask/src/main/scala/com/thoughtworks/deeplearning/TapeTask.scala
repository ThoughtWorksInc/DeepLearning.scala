package com.thoughtworks.deeplearning

import com.thoughtworks.raii.asynchronous.Do

import scalaz.concurrent.Future._
import scalaz.syntax.all._
import scalaz.concurrent.{Future, Task}
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.resourcet.ResourceT._
import com.thoughtworks.raii.resourcet.ResourceT
import com.thoughtworks.tryt.TryT

import scala.util.Try
import scalaz.\/
import scalaz.std.`try`.toDisjunction
object TapeTask {

  def predict[OutputData, OutputDelta](forward: Do[_ <: Tape.Aux[OutputData, OutputDelta]]): Task[OutputData] = {
    val Do(doOutputData) = forward.map(_.data)
    import com.thoughtworks.raii.resourcet.ResourceT._
    new Task(ResourceT.run(ResourceT(doOutputData).map(toDisjunction)))
  }

  def train[OutputData, OutputDelta](forward: Do[_ <: Tape.Aux[OutputData, OutputDelta]])(
      implicit trainable: Trainable[OutputData, OutputDelta]): Task[OutputData] = {

    val doOutputData: Do[OutputData] = forward.flatMap[OutputData] { outputTape =>
      Do.delay(outputTape.backward(trainable(outputTape.data))).map { _ =>
        outputTape.data
      }
    }

    val Do(futureOutputData) = doOutputData

    import com.thoughtworks.raii.resourcet.ResourceT._
    new Task(ResourceT.run(ResourceT(futureOutputData).map(toDisjunction)))
  }

  trait Trainable[-Data, +Delta] {
    def apply(data: Data): Do[_ <: Delta]
  }

}
