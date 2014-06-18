package com.qifun.statelessFuture.io

import java.io.InputStream
import java.nio.ByteBuffer
import scala.annotation.tailrec
import java.nio.BufferUnderflowException
import scala.collection.generic.Growable
import scala.collection.immutable.VectorBuilder
import scala.collection.mutable.ArrayBuffer

/**
 * @define This PagedInputStream
 */
private[io] class PagedInputStream(
  private[io] val buffers: scala.collection.mutable.Queue[ByteBuffer] = collection.mutable.Queue.empty[ByteBuffer])
  extends InputStream {

  override def available = buffers.foldLeft(0) { _ + _.remaining }

  override def read(): Int = {
    if (buffers.isEmpty) {
      -1
    } else {
      val buffer = buffers.front
      try {
        val result = buffer.get
        if (buffer.remaining == 0) {
          buffers.dequeue()
        }
        result
      } catch {
        case _: BufferUnderflowException =>
          -1
      }
    }

  }

  @tailrec
  private def read(b: Array[Byte], off: Int, len: Int, count: Int): Int = {
    if (buffers.isEmpty) {
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
      val buffer = buffers.front
      val remaining = buffer.remaining
      if (remaining > len) {
        buffer.get(b, off, len)
        count + len
      } else {
        buffer.get(b, off, remaining)
        buffers.dequeue()
        read(b, off + remaining, len - remaining, count + remaining)
      }
    }
  }

  @tailrec
  private def skip(len: Long, count: Long): Long = {
    if (buffers.isEmpty || len == 0) {
      count
    } else {
      val buffer = buffers.front
      val remaining = buffer.remaining
      if (remaining > len) {
        buffer.position(buffer.position + len.toInt)
        count + len
      } else {
        buffers.dequeue()
        skip(len - remaining, count + remaining)
      }
    }
  }

  override def skip(len: Long): Long = skip(len, 0)

  override final def read(b: Array[Byte]): Int = read(b, 0, b.length, 0)

  override def read(b: Array[Byte], off: Int, len: Int): Int = {
    read(b, off, len, 0)
  }


  @tailrec
  private def move(output: Growable[ByteBuffer], length: Long, count: Long): Long = {
    if (buffers.isEmpty || length == 0) {
      count
    } else {
      val buffer = buffers.front
      val remaining = buffer.remaining
      if (remaining > length) {
        val duplicated = buffer.duplicate()
        val newPosition = buffer.position + length.toInt
        buffer.position(newPosition)
        duplicated.limit(newPosition)
        output += duplicated
        count + length
      } else {
        output += buffers.dequeue()
        move(output, length - remaining, count + remaining)
      }
    }
  }

  /**
   * Read `length` bytes data from this [[$This]],
   * and append these data to `output`.
   * 
   * @return Number of bytes actually been processed, which may be less than `length`.
   */
  def move(output: Growable[ByteBuffer], length: Long): Long = {
    move(output, length, 0)
  }

}