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

package com.qifun.statelessFuture.util

import scala.collection.concurrent.TrieMap
import java.util.concurrent.atomic.AtomicReference
import scala.annotation.tailrec

object CancellationToken {

  final class Registration private[CancellationToken] (val id: Int) extends AnyVal

  private type Reference = AtomicReference[(Int, Map[Int, () => Unit])]

  @tailrec
  private final def unregister(reference: Reference, registration: Registration) {
    val oldState @ (nextId, handlers) = reference.get()
    if (!reference.compareAndSet(oldState, (nextId, handlers - registration.id))) {
      unregister(reference, registration)
    }

  }

  @tailrec
  private final def register(reference: Reference, handler: () => Unit): Registration = {
    val oldState @ (nextId, handlers) = reference.get()
    if (reference.compareAndSet(oldState, (nextId + 1, handlers + (nextId -> handler)))) {
      new Registration(nextId)
    } else {
      register(reference, handler)
    }
  }

}

final class CancellationToken(val reference: CancellationToken.Reference) extends AnyVal {

  final def unregister(registration: CancellationToken.Registration) =
    CancellationToken.unregister(reference, registration)

  final def register(handler: () => Unit) =
    CancellationToken.register(reference, handler)
} 