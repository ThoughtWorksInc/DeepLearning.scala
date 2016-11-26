package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.any.Type

// TODO: rename to sized

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object seq {

  type Seq[A <: Type[_, _]] = Type[scala.Seq[Type.DataOf[A]], (Int, Type.DeltaOf[A])]

}
