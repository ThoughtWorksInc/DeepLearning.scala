package com.thoughtworks.deeplearning

import com.qifun.statelessFuture.Future
import com.thoughtworks.deeplearning.Layer.Tape
import org.scalatest._
import DifferentiableFloat._
import com.qifun.statelessFuture.util.Promise
import com.thoughtworks.deeplearning.Symbolic.Layers.Literal

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DifferentiableFloatSpec extends AsyncFreeSpec with Matchers {

  private def toConcurrentFuture[A](statelessFuture: Future[A]): scala.concurrent.Future[A] = {
    val p = Promise[A]
    p.completeWith(statelessFuture)
    p
  }

  "xxxx" in toConcurrentFuture {
//
//    def layer[Input0, Output0 <: Tape](f:  Input0 => Future[Output0]):Layer.Aux[Input0, Output0] = new Layer {
//      override type Output = Output0
//      override type Input = Input0
//      override def forward(input: Input): Future[Output] = f(input)
//    }
//
//    val myNeuralNetwork2 = layer { input: Tape.Aux[Float, Float] => Future {
//        val t = input + input
//        if (t.value > 0.5f) {
//          t + input
//        } else {
//          t
//        }
//      }
//    }

    val myNeuralNetwork: Layer.Aux[Tape.Aux[Float, Float], Tape.Aux[Float, Float]] = new Layer {
      override def forward(input: Input): Future[Output] = Future {
        val t = input + input
        try {
          val t2 = (t + t)
          try {
            val t3 = t2 + t
            t3
          } finally {
            t2.close()
          }
        } finally {
          t.close()
        }
      }

      override type Output = Tape.Aux[Float, Float]
      override type Input = Tape.Aux[Float, Float]
    }

    Future {
      val l = Literal(1.0f)
      try {
        val outputTape = myNeuralNetwork.forward(l).await
        try {
          outputTape.backward(1.0f).await should be(1.0f)
        } finally {
          outputTape.close().await
        }
      } finally {
        l.close().await
      }
    }

  }

}
