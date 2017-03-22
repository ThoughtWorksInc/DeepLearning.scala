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
    val nd4jVersion = "0.8.0"
    Seq("org.nd4j" %% "nd4s" % nd4jVersion,
        "org.nd4j" % "nd4j-api" % nd4jVersion,
        "org.nd4j" % "nd4j-native-platform" % nd4jVersion)
  }
}
