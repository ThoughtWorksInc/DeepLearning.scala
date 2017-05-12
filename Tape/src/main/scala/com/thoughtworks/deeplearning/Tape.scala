package com.thoughtworks.deeplearning

import com.thoughtworks.raii.asynchronous.Do

import scalaz.{-\/, \/-}
import scalaz.concurrent.Future

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Tape[+Data, -Delta](data: Data, backward: Do[Delta] => Future[Unit])

object Tape {

  def literal[Data](data: Data) = Tape[Data, Any](data, Function.const(Future.now(())))

}
