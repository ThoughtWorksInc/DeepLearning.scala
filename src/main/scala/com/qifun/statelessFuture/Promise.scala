/*
 * stateless-future
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

import java.util.concurrent.atomic.AtomicReference
import scala.util.control.Exception.Catcher
import scala.annotation.tailrec
import scala.util.control.TailCalls._
import scala.util.Try
import scala.util.Failure
import scala.util.Success
import scala.Left
import scala.Right

object Promise {
  def apply[A]() = new Promise[A]

  private implicit class Scala210TailRec[A](underlying: TailRec[A]) {
    final def flatMap[B](f: A => TailRec[B]): TailRec[B] = {
      tailcall(f(underlying.result))
    }
  }

}

/**
 * The stateful variant that implement the API of Stateless Future. It's not a real Stateless Future, must be used very carefully!
 */
final class Promise[A] private (val state: AtomicReference[Either[List[(A => TailRec[Unit], Catcher[TailRec[Unit]])], Try[A]]] = new AtomicReference[Either[List[(A => TailRec[Unit], Catcher[TailRec[Unit]])], Try[A]]](Left(Nil))) extends AnyVal with StatefulFuture[A] { // TODO: 把List和Tuple2合并成一个对象，以减少内存占用

  // 为了能在Scala 2.10中编译通过
  import Promise.Scala210TailRec

  private def dispatch(handlers: List[(A => TailRec[Unit], Catcher[TailRec[Unit]])], value: Try[A]): TailRec[Unit] = {
    handlers match {
      case Nil => done(())
      case (body, catcher) :: tail => {
        (value match {
          case Success(a) => {
            body(a)
          }
          case Failure(e) => {
            if (catcher.isDefinedAt(e)) {
              catcher(e)
            } else {
              throw e
            }
          }
        }).flatMap { _ =>
          dispatch(tail, value)
        }
      }
    }
  }

  override final def value = state.get.right.toOption

  // @tailrec // Comment this because of https://issues.scala-lang.org/browse/SI-6574
  final def complete(value: Try[A]): TailRec[Unit] = {
    state.get match {
      case oldState @ Left(handlers) => {
        if (state.compareAndSet(oldState, Right(value))) {
          dispatch(handlers, value)
        } else {
          complete(value)
        }
      }
      case Right(origin) => {
        throw new IllegalStateException
      }
    }
  }

  // @tailrec // Comment this because of https://issues.scala-lang.org/browse/SI-6574
  override final def onComplete(body: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
    state.get match {
      case Right(value) => {
        value match {
          case Success(a) => {
            body(a)
          }
          case Failure(e) => {
            if (catcher.isDefinedAt(e)) {
              catcher(e)
            } else {
              throw e
            }
          }
        }
      }
      case oldState @ Left(tail) => {
        if (state.compareAndSet(oldState, Left((body, catcher) :: tail))) {
          done(())
        } else {
          onComplete(body)
        }
      }
    }
  }

}