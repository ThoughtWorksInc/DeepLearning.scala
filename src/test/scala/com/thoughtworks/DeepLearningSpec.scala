package com.thoughtworks

import com.thoughtworks.DeepLearning.{Array2D, PointfreeDeepLearning}

import scala.language.higherKinds
import Predef.{any2stringadd => _, _}
import PointfreeDeepLearning.ops._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
class DeepLearningSpec {
  final def addDoubleDouble[F[_] : PointfreeDeepLearning](left: F[Double], right: F[Double]): F[Double] = {
    left + right
  }

  final def addDoubleArray[F[_] : PointfreeDeepLearning](left: F[Double], right: F[Array2D]): F[Array2D] = {
    left + right
  }

  final def addArrayDouble[F[_] : PointfreeDeepLearning](left: F[Array2D], right: F[Double]): F[Array2D] = {
    left + right
  }

  final def addArrayArray[F[_] : PointfreeDeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D] = {
    left + right
  }


}
