addCompilerPlugin("com.thoughtworks.implicit-dependent-type" %% "implicit-dependent-type" % "2.0.0" % Test)

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

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
