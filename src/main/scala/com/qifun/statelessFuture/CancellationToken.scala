package com.qifun.statelessFuture.util

trait CancellationToken {
  def register(handler: () => Unit): AutoCloseable
} 