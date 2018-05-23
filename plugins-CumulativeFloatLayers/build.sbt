enablePlugins(Example)

exampleSuperTypes ~= { oldExampleSuperTypes =>
  import oldExampleSuperTypes._
  updated(indexOf("_root_.org.scalatest.FreeSpec"), "_root_.org.scalatest.AsyncFreeSpec")
}

exampleSuperTypes += "_root_.com.thoughtworks.deeplearning.scalatest.ThoughtworksFutureToScalaFuture"

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1" % Test

scalacOptions += "-Ypartial-unification"

addCompilerPlugin("com.thoughtworks.dsl" %% "compilerplugins-bangnotation" % "1.0.0-RC9")

addCompilerPlugin("com.thoughtworks.dsl" %% "compilerplugins-reseteverywhere" % "1.0.0-RC9")
