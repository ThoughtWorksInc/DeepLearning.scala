addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % Test

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1"

libraryDependencies += "com.thoughtworks.feature" %% "demixin" % "2.0.0-M0"

fork in Test := true

import Ordering.Implicits._

publishArtifact := {
  if (VersionNumber(scalaVersion.value).numbers >= Seq(2, 12)) {
    false
  } else {
    true
  }
}

skip in compile := {
  if (VersionNumber(scalaVersion.value).numbers >= Seq(2, 12)) {
    true
  } else {
    false
  }
}

libraryDependencies ++= {
  if (VersionNumber(scalaVersion.value).numbers >= Seq(2, 12)) {
    Nil
  } else {
    Seq("org.nd4j" %% "nd4s" % "0.7.2",
      "org.nd4j" % "nd4j-api" % "0.7.2",
      "org.nd4j" % "nd4j-native-platform" % "0.7.2" % Test)
  }
}