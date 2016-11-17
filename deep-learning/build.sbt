name := "differentiable"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

SbtNd4J.addNd4jRuntime(Test)

libraryDependencies += "org.typelevel" %% "cats" % "0.7.2"

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

addCompilerPlugin("com.milessabin" % "si2712fix-plugin" % "1.2.0" cross CrossVersion.full)

addCompilerPlugin("org.spire-math" % "kind-projector" % "0.8.2" cross CrossVersion.binary)
