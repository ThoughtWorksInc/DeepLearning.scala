package com.qifun.statelessFuture.io

import java.io.InputStream
import java.nio.ByteBuffer
import scala.annotation.tailrec

private[io] class PagedInputStream private[io] (
  protected val buffers: scala.collection.mutable.Queue[ByteBuffer] = collection.mutable.Queue.empty[ByteBuffer],
  protected var _available: Int = 0) extends InputStream {

  override final def available: Int = _available

  /**
   * @note 如果[[available_=]]返回的[[Future]]操作正在执行，将导致未定义行为。
   */
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

  final def duplicate(): PagedInputStream = {
    new PagedInputStream(
      buffers.map(_.duplicate),
      _available)
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
   * @note 如果[[available_=]]返回的[[Future]]操作正在执行，将导致未定义行为。
   */
  override final def skip(len: Long): Long = skip(len, 0)

  /**
   * @note 如果[[available_=]]返回的[[Future]]操作正在执行，将导致未定义行为。
   */
  override final def read(b: Array[Byte]): Int = read(b, 0, b.length, 0)

  /**
   * @note 如果[[available_=]]返回的[[Future]]操作正在执行，将导致未定义行为。
   */
  override final def read(b: Array[Byte], off: Int, len: Int): Int = {
    read(b, off, len, 0)
  }

  /**
   * @note Overriding this method in child class of [[SocketInputStream]] is forbidden.
   */
  override final def mark(readlimit: Int) { super.mark(readlimit) }

  /**
   * @note Overriding this method in child class of [[SocketInputStream]] is forbidden.
   */
  override final def markSupported() = super.markSupported()

  /**
   * @note Overriding this method in child class of [[SocketInputStream]] is forbidden.
   */
  override final def reset() { super.reset() }

}