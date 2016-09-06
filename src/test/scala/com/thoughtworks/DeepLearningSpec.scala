package com.thoughtworks

import org.scalatest._
import cats.implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DeepLearningSpec extends FreeSpec with Matchers with Inside {

  "-" in {
    import DeepLearning._

    val input = Input[scala.Double, scala.Double]
    val output = input - Double(3.0)




    //    network.forwardApply(Double(2.9))

  }

}