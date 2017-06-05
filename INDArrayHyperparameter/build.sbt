libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1"

libraryDependencies += "com.thoughtworks.feature" %% "the" % "2.0.0-RC5"

libraryDependencies += "com.thoughtworks.feature" %% "partialapply" % "2.0.0-RC5"

libraryDependencies += "com.thoughtworks.feature" %% "implicitapply" % "2.0.0-RC5"

libraryDependencies += "com.thoughtworks.feature" %% "byname" % "2.0.0-RC5"

enablePlugins(Example)

name in generateExample := "DifferentiableINDArraySpec"

addCompilerPlugin(("org.scalameta" % "paradise" % "3.0.0-M8").cross(CrossVersion.patch))

fork in Test := true

import Ordering.Implicits._

libraryDependencies ++= {
  if (VersionNumber(scalaVersion.value).numbers >= Seq(2, 12)) {
    Nil
  } else {
    Seq("org.nd4j" %% "nd4s" % "0.7.2",
        "org.nd4j" % "nd4j-api" % "0.7.2",
        "org.nd4j" % "nd4j-native-platform" % "0.7.2" % Test)
  }
}

scalacOptions += "-Ypartial-unification"
