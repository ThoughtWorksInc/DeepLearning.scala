package com.qifun.statelessFuture

package object util {

  type Poll[AwaitResult] = CancellablePromise[AwaitResult]

  type Sleep = CancellablePromise[Unit]

}