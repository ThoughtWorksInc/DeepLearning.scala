package com.qifun.statelessFuture

package object util {

  type Poll[AwaitResult] = CancellableFuture[AwaitResult]

  type Sleep = CancellableFuture[Unit]

  type JumpInto = Future.Stateless[Unit]

}
