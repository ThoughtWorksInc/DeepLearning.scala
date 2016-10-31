package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.any.ast.Identity

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object any {

  type Any = {
    type Data
    type Delta
  }

  def input[TypePair <: Any] = {
    Identity[Batch.FromTypePair[TypePair]]
  }

}
