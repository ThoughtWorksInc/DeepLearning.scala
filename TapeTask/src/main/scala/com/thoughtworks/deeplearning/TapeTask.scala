package com.thoughtworks.deeplearning

import com.thoughtworks.raii.future.Do

import scalaz.concurrent.Future._
import scalaz.syntax.all._
import scalaz.concurrent.{Future, Task}
import com.thoughtworks.raii.transformers.ResourceFactoryT.resourceFactoryTParallelApplicative
import com.thoughtworks.raii.future.Do.doParallelApplicative
import com.thoughtworks.raii.transformers.ResourceFactoryT.resourceFactoryTMonadError
import com.thoughtworks.raii.future.Do.doMonadErrorInstances
import com.thoughtworks.raii.transformers.ResourceFactoryT
import com.thoughtworks.tryt.TryT

import scala.util.Try
import scalaz.\/
import scalaz.std.`try`.toDisjunction
object TapeTask {

  def predict[OutputData, OutputDelta](forward: Do[_ <: Tape.Aux[OutputData, OutputDelta]]): Task[OutputData] = {
    val Do(doOutputData) = forward.map(_.data)
    import com.thoughtworks.raii.transformers.ResourceFactoryT._
    new Task(ResourceFactoryT.run(ResourceFactoryT(doOutputData).map(toDisjunction)))
  }

  def train[OutputData, OutputDelta](forward: Do[_ <: Tape.Aux[OutputData, OutputDelta]])(
      implicit trainable: Trainable[OutputData, OutputDelta]): Task[OutputData] = {

    val doOutputData: Do[OutputData] = forward.flatMap[OutputData] { outputTape =>
      Do.delay(outputTape.backward(trainable(outputTape.data))).map { _ =>
        outputTape.data
      }
    }

    val Do(futureOutputData) = doOutputData

    import com.thoughtworks.raii.transformers.ResourceFactoryT._
    new Task(ResourceFactoryT.run(ResourceFactoryT(futureOutputData).map(toDisjunction)))
  }

  trait Trainable[-Data, +Delta] {
    def apply(data: Data): Do[_ <: Delta]
  }

}
