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
package io

import scala.util.control.Exception.Catcher
import java.nio.channels._
import scala.annotation.tailrec
import java.io.InputStream
import java.io.EOFException
import java.nio.ByteBuffer
import java.util.concurrent.TimeUnit
import java.io.IOException
import com.dongxiguo.fastring.Fastring.Implicits._

object SocketInputStream {
  implicit private val (logger, formatter, appender) = ZeroLoggerFactory.newLogger(this)
  import formatter._

  final def apply(s: AsynchronousSocketChannel, t: Long, tu: TimeUnit, bs: Int = 1500) =
    new SocketInputStream {
      override protected final def readingTimeout: Long = t

      override protected final def readingTimeoutUnit: TimeUnit = tu

      override protected final val socket = s
    }

}

abstract class SocketInputStream extends InputStream {
  import SocketInputStream._
  import formatter._

  private val buffers = collection.mutable.Queue.empty[ByteBuffer]

  protected val socket: AsynchronousSocketChannel

  /**
   * 每次读取最少要准备多大的缓冲区
   */
  protected def minBufferSizePerRead = 1500

  protected def readingTimeout: Long

  protected def readingTimeoutUnit: TimeUnit

  override final def available: Int = _available

  final def capacity = buffers.foldLeft(0) { _ + _.remaining }

  private var _available: Int = _

  override final def read(): Int = {
    if (_available > 0) {
      if (buffers.isEmpty) {
        -1
      } else {
        val buffer = buffers.front
        val result = buffer.get
        if (buffer.remaining == 0) {
          buffers.dequeue()
        }
        _available -= 1
        result
      }
    } else {
      -1
    }
  }

  @tailrec
  private def read(b: Array[Byte], off: Int, len: Int, count: Int): Int = {
    if (_available <= 0 || buffers.isEmpty) {
      if (count == 0) {
        if (len == 0) {
          0
        } else {
          -1
        }
      } else {
        count
      }
    } else if (len == 0) {
      count
    } else {
      val l = math.min(len, _available)
      val buffer = buffers.front
      val remaining = buffer.remaining
      if (remaining > l) {
        buffer.get(b, off, l)
        _available = _available - l
        count + l
      } else {
        buffer.get(b, off, remaining)
        _available = _available - remaining
        buffers.dequeue()
        read(b, off + remaining, l - remaining, count + remaining)
      }
    }
  }

  @tailrec
  private def skip(len: Long, count: Long): Long = {
    if (_available <= 0 || buffers.isEmpty) {
      count
    } else if (len == 0) {
      count
    } else {
      val l = math.min(len, _available).toInt
      val buffer = buffers.front
      val remaining = buffer.remaining
      if (remaining > l) {
        buffer.position(buffer.position + l)
        _available = _available - l
        count + l
      } else {
        _available = _available - remaining
        buffers.dequeue()
        skip(len, count + remaining)
      }
    }
  }

  /**
   * @note 当 prepare 未完毕时调用skip将导致未定义行为。
   */
  override final def skip(len: Long): Long = skip(len, 0)

  /**
   * @note 当 prepare 未完毕时调用read将导致未定义行为。
   */
  override final def read(b: Array[Byte]): Int = read(b, 0, b.length, 0)

  /**
   * @note 当 prepare 未完毕时调用read将导致未定义行为。
   */
  override final def read(b: Array[Byte], off: Int, len: Int): Int = {
    read(b, off, len, 0)
  }

  /**
   * Close the associated socket.
   */
  override final def close() {
    socket.close()
  }

  override final def mark(readlimit: Int) { super.mark(readlimit) }

  override final def markSupported() = super.markSupported()

  override final def reset() { super.reset() }

  @throws(classOf[IOException])
  private def readChannel(
    bytesToRead: Long, buffers: Array[ByteBuffer],
    offset: Int, length: Int): Future[Unit] = Future {
    val n =
      Nio2.read(
        socket,
        buffers, offset, length,
        readingTimeout, readingTimeoutUnit).await
    if (n >= 0 && n < bytesToRead) {
      val newOffset = buffers.indexWhere(
        { buffer => buffer.hasRemaining },
        offset)
      readChannel(bytesToRead - n,
        buffers,
        newOffset,
        length - newOffset + offset).await
    }
  }

  @throws(classOf[IOException])
  private def readChannel(bytesToRead: Int, buffer: ByteBuffer): Future[Unit] = Future {
    val n =
      Nio2.read(socket, buffer, readingTimeout, readingTimeoutUnit).await
    if (n >= 0 && n < bytesToRead) {
      readChannel(bytesToRead - n, buffer).await
    }
  }

  @throws(classOf[IOException])
  private def externalRead(bytesToRead: Int): Future[Unit] = Future {
    val bufferSize = math.max(bytesToRead, minBufferSizePerRead)
    if (buffers.isEmpty) {
      val buffer = ByteBuffer.allocate(bufferSize)
      readChannel(bytesToRead, buffer).await
      buffer.flip()
      buffers.enqueue(buffer)
    } else {
      val buffer0 = {
        val t = buffers.last.duplicate
        t.mark()
        t.position(t.limit)
        t.limit(t.capacity)
        t.slice
      }
      if (bufferSize > buffer0.remaining) {
        val buffer1 = ByteBuffer.allocate(bufferSize - buffer0.remaining)
        readChannel(bytesToRead, Array(buffer0, buffer1), 0, 2).await
        buffer0.flip()
        buffers.enqueue(buffer0)
        buffer1.flip()
        buffers.enqueue(buffer0, buffer1)
      } else {
        readChannel(bytesToRead, buffer0).await
        buffer0.flip()
        buffers.enqueue(buffer0)
      }
    }
  }

  /**
   * Workaround to enable
   * <code>asynchronousInputStream.available = 123</code> syntax.
   */
  @inline final def available(implicit dummy: DummyImplicit): Int = available()

  /**
   * 准备若干个字节的数据。
   * @throws java.io.EOFException 如果对面已经断开连接，会触发本异常
   */
  @throws(classOf[EOFException])
  final def available_=(bytesRequired: Int): Future[Unit] = Future {
    logger.fine {
      fast"Bytes avaliable now: ${available.toString}, expected: ${bytesRequired.toString}"
    }
    val c = capacity
    if (bytesRequired > c) {
      logger.finest("Read from socket.")
      try {
        externalRead(bytesRequired - c).await
      } catch {
        case e: Exception =>
          _available = math.min(bytesRequired, capacity)
          logger.severe(e)
          throw e
      }
      val newCapacity = capacity
      if (bytesRequired > newCapacity) {
        _available = newCapacity
        val e = new EOFException
        logger.warning(e)
        throw e
      } else {
        _available = bytesRequired
      }
    } else {
      logger.finest("Bytes avaiable is enough. Don't read from socket.")
      _available = bytesRequired
    }
    logger.finer {
      fast"Bytes avaiable is ${_available.toString} now."
    }
  }

}
