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
import scala.util.Failure
import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls._

final case class Zip[ThisAwaitResult, ThatAwaitResult](
  thisFuture: Future.Stateful[ThisAwaitResult],
  thatFuture: Future.Stateful[ThatAwaitResult]) extends Future.Stateful[(ThisAwaitResult, ThatAwaitResult)] {

  override final def value = {
    (thisFuture.value, thatFuture.value) match {
      case (Some(Success(thisSuccess)), Some(Success(thatSuccess))) => Some(Success((thisSuccess, thatSuccess)))
      case (Some(Failure(e)), _) => Some(Failure(e))
      case (_, Some(Failure(e))) => Some(Failure(e))
      case _ => None
    }
  } 

  override final def onComplete(handler: ((ThisAwaitResult, ThatAwaitResult)) => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
    thisFuture.onComplete { thisSuccess =>
      thatFuture.onComplete { thatSuccess =>
        handler((thisSuccess, thatSuccess))
      }
    }
  }

}
