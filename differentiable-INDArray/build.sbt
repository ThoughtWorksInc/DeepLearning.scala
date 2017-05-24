addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % Test

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1"

libraryDependencies += "com.thoughtworks.feature" %% "demixin" % "2.0.0-RC2"

addCompilerPlugin(("org.scalameta" % "paradise" % "3.0.0-M8").cross(CrossVersion.patch))
libraryDependencies += "com.thoughtworks.example" %% "example" % "1.0.0"
sourceGenerators in Test += Def.task {
  val className = s"${name.value}Spec"
  val outputFile = (sourceManaged in Test).value / "ScaladocSpec.scala"
  val fileNames = (unmanagedSources in Compile).value
    .map { file =>
      import scala.reflect.runtime.universe._
      Literal(Constant(file.toString))
    }
    .mkString(",")
  val fileContent = raw"""
    package com.thoughtworks.deeplearning.differentiable;
    @_root_.com.thoughtworks.example($fileNames) class ScaladocSpec extends org.scalatest.AsyncFreeSpec
  """
  IO.write(outputFile, fileContent, scala.io.Codec.UTF8.charSet)
  Seq(outputFile)
}.taskValue
incOptions in Test := (incOptions in Test).value.withRecompileOnMacroDef(true)

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

scalacOptions += "-Ypartial-unification"
