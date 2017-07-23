enablePlugins(Example)

exampleSuperTypes ~= { oldExampleSuperTypes =>
  import oldExampleSuperTypes._
  updated(indexOf("_root_.org.scalatest.FreeSpec"), "_root_.org.scalatest.AsyncFreeSpec")
}

exampleSuperTypes += "_root_.com.thoughtworks.deeplearning.scalatest.ThoughtworksFutureToScalaFuture"

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1" % Test

libraryDependencies ++= {
  import Ordering.Implicits._
  if (VersionNumber(scalaVersion.value).numbers >= Seq(2, 12)) {
    Nil
  } else {
    Seq("org.nd4j" %% "nd4s" % "0.8.0",
        "org.nd4j" % "nd4j-api" % "0.8.0",
        "org.nd4j" % "nd4j-native-platform" % "0.8.0" % Test)
  }
}

fork in Test := true
