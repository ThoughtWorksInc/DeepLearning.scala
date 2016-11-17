package com.thoughtworks.deepLearning

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait BatchId {
  type Open <: Batch
  def open(): Open
}

object BatchId {
  type Aux[+Open0 <: Batch] = BatchId {
    type Open <: Open0
  }
}
