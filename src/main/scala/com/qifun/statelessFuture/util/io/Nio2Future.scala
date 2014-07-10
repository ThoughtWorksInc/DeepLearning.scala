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

import java.nio.channels.AsynchronousServerSocketChannel
import java.nio.channels.AsynchronousSocketChannel
import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls._
import java.nio.channels.CompletionHandler
import java.nio.ByteBuffer
import java.nio.channels.AsynchronousByteChannel
import java.util.concurrent.TimeUnit
import java.nio.channels.AsynchronousFileChannel
import java.nio.channels.FileLock
import java.net.SocketAddress
import java.nio.channels.ShutdownChannelGroupException

final case class Nio2Future[A](
  val underlying: CompletionHandler[A, Null] => Unit) extends AnyVal with Future.Stateless[A] {

  override final def onComplete(
    handler: A => TailRec[Unit])(
      implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
    try {
      underlying(new Nio2Future.HandlerToCompletionHandler(handler))
    } catch {
      case e if catcher.isDefinedAt(e) => catcher(e)
    }
    done(())
  }

}

object Nio2Future {

  private final class HandlerToCompletionHandler[A](
    handler: A => TailRec[Unit])(
      implicit catcher: Catcher[TailRec[Unit]])
    extends CompletionHandler[A, Null] {
    override final def completed(a: A, unused: Null) {
      handler(a).result
    }
    override final def failed(throwable: Throwable, unused: Null) {
      catcher.applyOrElse(throwable, { e: Throwable => throw e }).result
    }
  }

  final def lock(fd: AsynchronousFileChannel) =
    Nio2Future[FileLock] { fd.lock(null, _) }

  final def lock(fd: AsynchronousFileChannel, position: Long, size: Long, shared: Boolean) =
    Nio2Future[FileLock] { fd.lock(position, size, shared, null, _) }

  final def accept(serverSocket: AsynchronousServerSocketChannel) =
    Nio2Future[AsynchronousSocketChannel] { completionHandler =>
      try {
        serverSocket.accept(null, completionHandler)
      } catch {
        case e: ShutdownChannelGroupException =>
        // An IOException is passed to completionHandler.
        // Do nothing here, since you may not want to invoke the catcher twice.
      }
    }

  final def connect(socket: AsynchronousSocketChannel, address: SocketAddress) =
    Nio2Future[Void] { socket.connect(address, null, _) }

  final def read(fd: AsynchronousFileChannel, position: Long, buffer: ByteBuffer) =
    Nio2Future[Integer] { fd.read(buffer, position, null, _) }

  final def read(socket: AsynchronousByteChannel, buffer: ByteBuffer) =
    Nio2Future[Integer] { socket.read(buffer, null, _) }

  final def read(socket: AsynchronousSocketChannel, buffer: ByteBuffer, timeout: Long, unit: TimeUnit) =
    Nio2Future[Integer] { socket.read(buffer, timeout, unit, null, _) }

  final def read(
    socket: AsynchronousSocketChannel,
    buffer: Array[ByteBuffer],
    offset: Int,
    length: Int,
    timeout: Long,
    unit: TimeUnit) =
    Nio2Future[java.lang.Long] { socket.read(buffer, offset, length, timeout, unit, null, _) }

  final def write(fd: AsynchronousFileChannel, position: Long, buffer: ByteBuffer) =
    Nio2Future[Integer] { fd.write(buffer, position, null, _) }

  final def write(socket: AsynchronousByteChannel, buffer: ByteBuffer) =
    Nio2Future[Integer] { socket.write(buffer, null, _) }

  final def write(socket: AsynchronousSocketChannel, buffer: ByteBuffer, timeout: Long, unit: TimeUnit) =
    Nio2Future[Integer] { socket.write(buffer, timeout, unit, null, _) }

  final def write(
    socket: AsynchronousSocketChannel,
    buffer: Array[ByteBuffer],
    offset: Int,
    length: Int,
    timeout: Long,
    unit: TimeUnit) =
    Nio2Future[java.lang.Long] { socket.write(buffer, offset, length, timeout, unit, null, _) }

}