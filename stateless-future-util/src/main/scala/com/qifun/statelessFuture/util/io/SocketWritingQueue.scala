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
package io

import java.io.IOException
import java.nio.ByteBuffer
import java.nio.channels._
import java.nio.channels.InterruptedByTimeoutException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import scala.annotation.tailrec
import com.qifun.statelessFuture.Future
import scala.util.control.TailCalls._
import scala.util.control.Exception.Catcher
import scala.concurrent.duration.Duration

private object SocketWritingQueue {
  implicit private val (logger, formatter, appender) = ZeroLoggerFactory.newLogger(this)
  import formatter._

  private[SocketWritingQueue] sealed abstract class State

  private final case class Idle(
    val buffers: List[ByteBuffer]) extends State

  private final case class Running(
    val buffers: List[ByteBuffer]) extends State

  private final case class Closing(
    val buffers: List[ByteBuffer]) extends State

  /**
   * 表示[[socket.close]]已经调用过
   */
  private case object Closed extends State

  private case object Interrupted extends State

}

/**
 * 写入Socket的缓冲数据队列。
 *
 * @note 本[[SocketWritingQueue]]保证线程安全，允许多个线程向这里提交要写入的数据。
 * [[enqueue]]提交的缓冲区将按提交顺序排队写入，而不会混合。
 */
trait SocketWritingQueue {

  import SocketWritingQueue.logger
  import SocketWritingQueue.formatter
  import SocketWritingQueue.appender

  private val state = new AtomicReference[SocketWritingQueue.State](SocketWritingQueue.Idle(Nil))

  protected val socket: AsynchronousSocketChannel

  protected def writingTimeout: Duration

  private def writingTimeoutLong: Long = {
    if (writingTimeout.isFinite) {
      writingTimeout.length match {
        case 0L => throw new IllegalArgumentException("writingTimeout must not be zero!")
        case l => l
      }
    } else {
      0L
    }
  }

  private def writingTimeoutUnit: TimeUnit = {
    if (writingTimeout.isFinite) {
      writingTimeout.unit
    } else {
      TimeUnit.SECONDS
    }
  }

  @tailrec
  final def flush() {
    val oldState = state.get
    oldState match {
      case SocketWritingQueue.Idle(Nil) =>
      case SocketWritingQueue.Idle(buffers) =>
        if (state.compareAndSet(oldState, SocketWritingQueue.Running(Nil))) {
          startWriting(buffers.reverseIterator.toArray)
        } else {
          // retry
          flush()
        }
      case _ =>
    }
  }

  /**
   * 强行关闭本[[SocketWritingQueue]]。
   *
   * 立即关闭[[socket]]，抛弃所有尚未发送的数据。
   * 如果多次调用[[interrupt]]，只有第一次调用有效，后面几次会被忽略。
   */
  final def interrupt() {
    state.getAndSet(SocketWritingQueue.Interrupted) match {
      case SocketWritingQueue.Closed | SocketWritingQueue.Interrupted =>
      case _ =>
        socket.close()
        logger.fine("Socket " + socket + " is closed by interrupt().")
    }
  }

  /**
   * 优雅的关闭本[[SocketWritingQueue]]。
   *
   * 如果本[[SocketWritingQueue]]队列中存在尚未发送的数据，
   * 那么只有当这些数据全部交给[[socket]]发送后，[[socket]]才会真正被关闭。
   * 
   * 如果多次调用[[shutDown]]，只有第一次调用有效，后面几次会被忽略。
   */
  @tailrec
  final def shutDown() {
    val oldState = state.get
    oldState match {
      case SocketWritingQueue.Idle(Nil) =>
        if (state.compareAndSet(oldState, SocketWritingQueue.Closed)) {
          socket.close()
          logger.fine("No data to send. Socket " + socket + " is closed.")
        } else {
          // retry
          shutDown()
        }
      case SocketWritingQueue.Idle(buffers) =>
        val newState = SocketWritingQueue.Closing(Nil)
        if (state.compareAndSet(oldState, newState)) {
          val bufferArray = buffers.reverseIterator.toArray
          startWriting(bufferArray)
        } else {
          // retry
          shutDown()
        }
      case SocketWritingQueue.Running(buffers) =>
        val newState = SocketWritingQueue.Closing(Nil)
        if (!state.compareAndSet(oldState, newState)) {
          // retry
          shutDown()
        }
      case SocketWritingQueue.Interrupted |
        SocketWritingQueue.Closing(_) |
        SocketWritingQueue.Closed =>
    }
  }

  private val writeHandler = new CompletionHandler[java.lang.Long, Function1[Long, Unit]] {
    override final def completed(
      bytesWritten: java.lang.Long,
      continue: Function1[Long, Unit]) {
      continue(bytesWritten.longValue)
    }

    override final def failed(
      throwable: Throwable,
      continue: Function1[Long, Unit]) {
      if (throwable.isInstanceOf[IOException]) {
        interrupt()
      } else {
        throw throwable
      }
    }
  }

  private def writeChannel(buffers: Array[ByteBuffer]) = {
    Nio2Future.write(
      socket,
      buffers,
      0,
      buffers.length,
      writingTimeoutLong,
      writingTimeoutUnit)
  }

  @tailrec
  private final def writeMore(remainingBuffers: Iterator[ByteBuffer]) {
    val oldState = state.get
    oldState match {
      case SocketWritingQueue.Running(Nil) =>
        if (remainingBuffers.isEmpty) {
          if (!state.compareAndSet(oldState, SocketWritingQueue.Idle(Nil))) {
            // retry
            writeMore(remainingBuffers)
          }
        } else {
          startWriting(remainingBuffers.toArray)
        }
      case SocketWritingQueue.Running(buffers) =>
        if (state.compareAndSet(oldState, SocketWritingQueue.Running(Nil))) {
          startWriting((remainingBuffers ++ buffers.reverseIterator).toArray)
        } else {
          // retry
          writeMore(remainingBuffers)
        }
      case SocketWritingQueue.Closing(Nil) =>
        if (remainingBuffers.isEmpty) {
          if (state.compareAndSet(oldState, SocketWritingQueue.Closed)) {
            socket.close()
            logger.fine("Socket " + socket + " is closed after all data been sent.")
          } else {
            // retry
            writeMore(remainingBuffers)
          }
        } else {
          startWriting(remainingBuffers.toArray)
        }
      case SocketWritingQueue.Closing(buffers) =>
        if (state.compareAndSet(oldState, SocketWritingQueue.Closing(Nil))) {
          startWriting((remainingBuffers ++ buffers.reverseIterator).toArray)
        } else {
          // retry
          writeMore(remainingBuffers)
        }
      case SocketWritingQueue.Idle(_) | SocketWritingQueue.Closed =>
        throw new IllegalStateException
      case SocketWritingQueue.Interrupted =>
    }
  }

  private def startWriting(buffers: Array[ByteBuffer]) {
    implicit def catcher: Catcher[Unit] = {
      case e: IOException =>
        // 本身不处理，关闭socket通知读线程来处理
        interrupt()
    }
    for (bytesWritten <- writeChannel(buffers)) {
      val nextIndex = buffers indexWhere { _.hasRemaining }
      val remainingBuffers = nextIndex match {
        case -1 => Iterator.empty
        case nextIndex =>
          buffers.view(nextIndex, buffers.length).iterator
      }
      writeMore(remainingBuffers)
    }
  }

  /**
   * Add `buffers` to this [[SocketWritingQueue]].
   *
   * If this [[SocketWritingQueue]] is closing or closed,
   * the enqueue operation will be ignored.
   * 
   * @note This [[SocketWritingQueue]] will change `position` in each of `buffers`,
   * but will not change the content of these `buffers`.
   * To prevent this behavior, please `duplicate` these `buffers` before [[enqueue]].
   */
  @tailrec
  final def enqueue(buffers: ByteBuffer*) {
    val oldState = state.get
    oldState match {
      case SocketWritingQueue.Idle(oldBuffers) =>
        val newState = SocketWritingQueue.Idle(
          buffers.foldLeft(oldBuffers) { (oldBuffers, newBuffer) =>
            newBuffer :: oldBuffers
          })
        if (!state.compareAndSet(oldState, newState)) {
          // retry
          enqueue(buffers: _*)
        }
      case SocketWritingQueue.Running(oldBuffers) =>
        val newState = SocketWritingQueue.Running(
          buffers.foldLeft(oldBuffers) { (oldBuffers, newBuffer) =>
            newBuffer :: oldBuffers
          })
        if (!state.compareAndSet(oldState, newState)) {
          // retry
          enqueue(buffers: _*)
        }
      case SocketWritingQueue.Interrupted |
        SocketWritingQueue.Closing(_) |
        SocketWritingQueue.Closed =>
    }
  }

  /**
   * Add `buffer` to this [[SocketWritingQueue]].
   *
   * If this [[SocketWritingQueue]] is closing or closed,
   * the enqueue operation will be ignored.
   *
   * @note This [[SocketWritingQueue]] will change `position` in each of `buffers`,
   * but will not change the content of these `buffers`.
   * To prevent this behavior, please `duplicate` these `buffers` before [[enqueue]].
   */
  @tailrec
  final def enqueue(buffer: ByteBuffer) {
    val oldState = state.get
    oldState match {
      case SocketWritingQueue.Idle(buffers) =>
        val newState = SocketWritingQueue.Idle(buffer :: buffers)
        if (!state.compareAndSet(oldState, newState)) {
          // retry
          enqueue(buffer)
        }
      case SocketWritingQueue.Running(buffers) =>
        val newState = SocketWritingQueue.Running(buffer :: buffers)
        if (!state.compareAndSet(oldState, newState)) {
          // retry
          enqueue(buffer)
        }
      case SocketWritingQueue.Interrupted |
        SocketWritingQueue.Closing(_) |
        SocketWritingQueue.Closed =>
    }
  }
}
// vim: et sts=2 sw=2
