package com.qifun.statelessFuture
package util
package io

import java.nio.ByteBuffer
import scala.collection.generic.Growable
import java.io.InputStream

/**
 * @define This LimitablePagedInputStream
 */
private[io] trait LimitablePagedInputStream extends InputStream {

  private[io] var limit: Int = 0

  override final def available = limit

  final def capacity = super.available

  override abstract final def read(): Int = {
    if (limit > 0) {
      super.read() match {
        case -1 => -1
        case notEof => {
          limit -= 1
          notEof
        }
      }
    } else {
      -1
    }
  }

  override final def skip(len: Long): Long = {
    val result = super.skip(math.min(len, limit))
    // 因 math.min(len, limit) <= limit，
    // 且 result <= math.min(len, limit)
    // 所以可以断定 result <= limit
    // 又因limit是个Int，所以result一定也能用Int表示，
    // 所以，这里的toInt不会溢出。
    limit -= result.toInt
    result
  }

  override final def read(output: Array[Byte], offset: Int, length: Int): Int = {
    super.read(output, offset, math.min(length, limit)) match {
      case 0 if length > 0 => {
        -1
      }
      case result => {
        limit -= result
        result
      }
    }
  }

  override final def read(b: Array[Byte]): Int = {
    read(b, 0, b.length)
  }


}