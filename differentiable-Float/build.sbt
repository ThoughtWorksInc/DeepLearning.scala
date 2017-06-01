libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % Test

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1" % Test

libraryDependencies += "org.scalaz" %% "scalaz-effect" % "7.2.11" % Test

libraryDependencies += "org.slf4j" % "jul-to-slf4j" % "1.7.25" % Test

libraryDependencies += "org.slf4j" % "slf4j-api" % "1.7.25" % Test

libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.2" % Test

libraryDependencies += "ch.qos.logback" % "logback-core" % "1.2.2" % Test

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

addCompilerPlugin("org.spire-math" %% "kind-projector" % "0.9.3")

libraryDependencies += "com.thoughtworks.feature" %% "constructor" % "2.0.0-RC4"

libraryDependencies += "com.thoughtworks.feature" %% "new" % "2.0.0-RC4"

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