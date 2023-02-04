enablePlugins(Example)

import scala.meta._

exampleSuperTypes := exampleSuperTypes.value.map {
  case ctor"_root_.org.scalatest.FreeSpec" =>
    ctor"_root_.org.scalatest.AsyncFreeSpec"
  case otherTrait =>
    otherTrait
}

exampleSuperTypes += ctor"_root_.com.thoughtworks.deeplearning.scalatest.ThoughtworksFutureToScalaFuture"

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.2" % Test

libraryDependencies += "com.thoughtworks.feature" %% "mixins-implicitssingleton" % "2.3.0"

fork in Test := true
