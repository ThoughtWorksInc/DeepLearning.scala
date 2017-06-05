package com.thoughtworks.deeplearning

import com.thoughtworks.raii.asynchronous.Do

import scalaz.{-\/, \/-}
import scalaz.concurrent.Future

// type Layer[A] = Do[Tape[A, A]]
/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final case class Tape[+Data, -Delta](data: Data, backward: Do[Delta] => Future[Unit])

