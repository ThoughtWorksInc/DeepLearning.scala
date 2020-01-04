enablePlugins(Example)

import scala.meta._

exampleSuperTypes := exampleSuperTypes.value.map {
  case ctor"_root_.org.scalatest.FreeSpec" =>
    ctor"_root_.org.scalatest.AsyncFreeSpec"
  case otherTrait =>
    otherTrait
}

exampleSuperTypes += ctor"_root_.com.thoughtworks.deeplearning.scalatest.ThoughtworksFutureToScalaFuture"

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1" % Test

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

scalacOptions += "-Ypartial-unification"

libraryDependencies += "com.thoughtworks.feature" %% "mixins-implicitssingleton" % "2.1.0" % Test

addCompilerPlugin("com.thoughtworks.dsl" %% "compilerplugins-bangnotation" % "1.0.0-2-ddbc1c14")

addCompilerPlugin("com.thoughtworks.dsl" %% "compilerplugins-reseteverywhere" % "1.0.0-2-ddbc1c14")
