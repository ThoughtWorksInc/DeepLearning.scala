libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

SbtNd4J.addNd4jRuntime(Test)

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)
