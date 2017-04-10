package com.thoughtworks

import com.thoughtworks.raii.ResourceFactoryT

import scalaz.EitherT
import scalaz.concurrent.Future

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object deeplearning {

  type FutureResourceFactory[Result] = ResourceFactoryT[Future, Result]

  /** A computational node that computes a `Result`.
    *
    * The features of this `Compute` are accessible via [[scalaz.MonadError]] and [[scalaz.Nondeterminism]] type classes
    * that support RAII, exception handling and asynchronous computing.
    */
  type Compute[Result] = EitherT[FutureResourceFactory, Throwable, Result]

}
