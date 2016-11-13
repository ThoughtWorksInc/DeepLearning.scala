package com.thoughtworks.deepLearning
//
//import com.thoughtworks.deepLearning.NeuralNetwork._
//import com.thoughtworks.deepLearning.any.Any
//import cats.Eval
//import com.thoughtworks.deepLearning.array2D.ast.{Dot, Negative}
//import com.thoughtworks.deepLearning.boolean.ast.If

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
package object boolean {

  /** @template */
  type Boolean = utilities.Boolean
//
//  implicit final class BooleanOps[Input <: Batch](differentiable: NeuralNetwork.Aux[Input, Boolean#Batch]) {
//
//    def `if`[ThatInput <: Input, Output <: Batch](`then`: NeuralNetwork.Aux[ThatInput, Output])(
//        `else`: NeuralNetwork.Aux[ThatInput, Output]): NeuralNetwork.Aux[ThatInput, Output] = {
//      If[ThatInput, Output](differentiable, `then`, `else`)
//    }
//
//  }

}
