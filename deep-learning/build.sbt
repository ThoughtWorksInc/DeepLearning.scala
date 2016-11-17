name := "differentiable"

import org.apache.commons.lang3.SystemUtils

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

libraryDependencies += "org.nd4j" %% "nd4s" % "0.4-rc3.8"

libraryDependencies += "org.nd4j" % "nd4j-api" % "0.4-rc3.9"

SbtNd4J.addNd4jRuntime(Test)

libraryDependencies += "org.typelevel" %% "cats" % "0.7.2"

libraryDependencies += "com.chuusai" %% "shapeless" % "2.3.2"

libraryDependencies += "com.thoughtworks.extractor" %% "extractor" % "1.1.1"

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

addCompilerPlugin("com.milessabin" % "si2712fix-plugin" % "1.2.0" cross CrossVersion.full)

addCompilerPlugin("org.spire-math" % "kind-projector" % "0.8.2" cross CrossVersion.binary)
