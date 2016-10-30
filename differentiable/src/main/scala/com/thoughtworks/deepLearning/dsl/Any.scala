package com.thoughtworks.deepLearning
package dsl


/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait Any {
  type Data
  type Delta

  type Batch >: NeuralNetwork.Batch.Aux[Data, Delta] <: NeuralNetwork.Batch.Aux[Data, Delta]

  // TODO: test for factory

}
