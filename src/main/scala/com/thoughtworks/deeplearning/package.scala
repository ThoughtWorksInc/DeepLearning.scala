package com.thoughtworks

import com.thoughtworks.raii.ResourceFactoryT

import scalaz.EitherT
import scalaz.concurrent.Future

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object deeplearning {

  type FutureResourceFactory[A] = ResourceFactoryT[Future, A]

  type Compute[A] = EitherT[FutureResourceFactory, Throwable, A]

}
