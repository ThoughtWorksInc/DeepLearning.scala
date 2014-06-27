/*
 * stateless-future-util
 * Copyright 2014 深圳岂凡网络有限公司 (Shenzhen QiFun Network Corp., LTD)
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.qifun.statelessFuture
package util

import scala.util.Success
import scala.util.Try
import scala.util.Failure
import scala.util.control.Exception.Catcher

object Blocking {

  final def blockingAwait[A](future: Future[A]): A = {
    val lock = new AnyRef
    lock.synchronized {
      @volatile var result: Option[Try[A]] = None
      implicit def catcher: Catcher[Unit] = {
        case e: Exception => {
          lock.synchronized {
            result = Some(Failure(e))
            lock.notifyAll()
          }
        }
      }
      future.foreach { u =>
        lock.synchronized {
          result = Some(Success(u))
          lock.notify()
        }
      }
      while (result == None) {
        lock.wait()
      }
      val Some(some) = result
      some match {
        case Success(u) => u
        case Failure(e) => throw e
      }
    }
  }
}